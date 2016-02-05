#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/interrupt.h>
#include <linux/gpio.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/time.h>
#include <linux/timer.h>
#include <linux/types.h>
#include <linux/suspend.h>
#include <linux/pm.h>
#include <linux/pm_wakeup.h>
#include <linux/pm_wakeirq.h>
#include <linux/workqueue.h>
#include <asm/uaccess.h>

#define DEVICE_NAME "pir"
#define IR_GPIO_PORT 28

const static unsigned long timeout_sec = 5*HZ;

/* device callbacks */
static int device_open(struct inode *, struct file *);
static int device_release(struct inode *, struct file *);
static ssize_t device_read(struct file *, char *, size_t, loff_t *);
static ssize_t device_write(struct file *, const char *, size_t, loff_t *);

/* system syspend callbacks */
static int suspend_valid(suspend_state_t state);
static int suspend_enter(suspend_state_t state);

/* device power management callbacks */
static int device_on_prepare(struct device *dev);
static int device_on_suspend(struct device *dev);
static int device_on_resume(struct device *dev);

/* timer helper functions */
static void pir_timer_register(struct timer_list *ptimer, unsigned long timeover);
static void pir_timer_timeover(unsigned long arg);
static void pir_timer_delete(struct timer_list *ptimer);

static struct class *device_class;
static struct device *pir_device;
static dev_t devdev;

static int major_num;
static int is_device_open = 0;
static char event_detected = 0;
static char waiting_bit = 0;

static int pir_int_num;

static struct timer_list pir_timer;

static struct file_operations fops = {
    .read = device_read,
    .write = device_write,
    .open = device_open,
    .release = device_release
};

static struct platform_suspend_ops sops = {
    .valid = suspend_valid,
    .enter = suspend_enter
};

static struct dev_pm_ops dpmops = {
    .prepare = device_on_prepare,
    .suspend = device_on_suspend,
    .resume = device_on_resume
};

static struct dev_pm_domain dpmdomain;

static void suspend_func(struct work_struct *dummy)
{
    int ret;
    if ((ret = pm_suspend(PM_SUSPEND_MEM)) < 0)
    {
        printk("pm_suspend failed, ret=%d\n", ret);
    }
}

static DECLARE_WORK(suspend_work, suspend_func);

static int suspend_valid(suspend_state_t state)
{
    return (state == PM_SUSPEND_MEM);
}

static int device_on_prepare(struct device *dev)
{
    printk("device_on_suspend\n");

    return 0;
}

static int device_on_suspend(struct device *dev)
{
    int ret;

    printk("device_on_prepare\n");

    ret = enable_irq_wake(pir_int_num);
    if (ret < 0)
    {
        printk("ERROR at enable_irq_wake: error code %d\n", ret);
        return ret;
    }

    return 0;
}

static int device_on_resume(struct device *dev)
{
    printk("device_on_resume\n");
    disable_irq_wake(pir_int_num);
    return 0;
}

static int suspend_enter(suspend_state_t state)
{
    return 0;
}

static void wakeup_func(struct work_struct *dummy)
{
    printk("Wakeup!\n");
}

static DECLARE_WORK(wakeup_work, wakeup_func);

irqreturn_t irq_handler(int irq, void *dev_id)
{
    static int initial_event = 1;

    event_detected = !event_detected;
    if (event_detected)
    {
        schedule_work(&wakeup_work);
        waiting_bit = 1;
        pir_timer_delete(&pir_timer);
        if (!initial_event)
            pm_wakeup_event(pir_device, 0);
        initial_event = 0;
    }
    if (waiting_bit && !event_detected)
    {
        pir_timer_register(&pir_timer, timeout_sec);
    }

    return IRQ_HANDLED;
}

static int __init init_pir_module(void)
{
    int ret;

    /* character device creation */
    major_num = register_chrdev(0, DEVICE_NAME, &fops);
    if (major_num < 0)
    {
        printk("ERROR: registering char device %s failed with %d\n", DEVICE_NAME, major_num);
        return major_num;
    }

    device_class = class_create(THIS_MODULE, DEVICE_NAME);
    if (device_class == NULL)
    {
        printk("ERROR: class create error\n");
        return -1;
    }
    device_class->pm = &dpmops;

    devdev = MKDEV(major_num, 0);
    pir_device = device_create(device_class, NULL, devdev, NULL, DEVICE_NAME);

    /* PIR GPIO registration */
    ret = gpio_request(IR_GPIO_PORT, "IR GPIO");
    if (ret < 0)
    {
        printk("ERROR: cannot request GPIO %d: error code %d\n", IR_GPIO_PORT, ret);
        return ret;
    }
    gpio_direction_input(IR_GPIO_PORT);

    pir_int_num = gpio_to_irq(IR_GPIO_PORT);
    if (pir_int_num < 0)
    {
        printk("ERROR: cannot request interrupt for GPIO %d: error code %d\n", IR_GPIO_PORT, pir_int_num);
        return pir_int_num;
    }

    printk("Interrupt for GPIO %d: %d\n", IR_GPIO_PORT, pir_int_num);

    /* ISR registration */
    ret = request_irq(pir_int_num,
            irq_handler,
            IRQF_TRIGGER_RISING | IRQF_TRIGGER_FALLING,
            "PIR IRQ handler",
            NULL);

    if (ret != 0)
    {
        printk("ERROR: cannot request IRQ %d, error code %d\n", pir_int_num, ret);
        return ret;
    }

    /* device wakeup setting */
/*    ret = device_init_wakeup(pir_device, true);
    if (ret != 0)
    {
        printk("ERROR at device_init_wakeup: error code %d\n", ret);
        return ret;
    }

    ret = dev_pm_set_wake_irq(pir_device, pir_int_num);
    if (ret != 0)
    {
        printk("ERROR at dev_pm_set_wake_irq: error code %d\n", ret);
        return ret;
    }

    if (!device_may_wakeup(pir_device))
    {
        printk("???\n");
        return -1;
    }*/

    dpmdomain.ops = dpmops;
    pir_device->pm_domain = &dpmdomain;

    /* suspend opeartions setup */
    suspend_set_ops(&sops);

    return 0;
}

static void __exit exit_pir_module(void)
{
    if (pir_device)
    {
        device_init_wakeup(pir_device, false);
    }

    pir_timer_delete(&pir_timer);
    free_irq(pir_int_num, NULL);
    gpio_free(IR_GPIO_PORT);
    device_destroy(device_class, devdev);
    class_destroy(device_class);
    unregister_chrdev(major_num, DEVICE_NAME);
}

static int device_open(struct inode *inode, struct file *file)
{
    if (is_device_open)
    {
        return -EBUSY;
    }

    is_device_open++;
    try_module_get(THIS_MODULE);
    return 0;
}

static int device_release(struct inode *inode, struct file *file)
{
    is_device_open--;
    module_put(THIS_MODULE);
    return 0;
}

static ssize_t device_read(struct file *filp, char *buffer, size_t length, loff_t *offset)
{
    int bytes_read = 0;
    int bytes_to_read = 2;

    while (length && bytes_to_read)
    {
        put_user(event_detected, buffer++);
        bytes_to_read--;
        length--;
        
        put_user(waiting_bit, buffer++);
        bytes_to_read--;
        length--;
    }

    return bytes_read;
}

static ssize_t device_write(struct file *filp, const char *buffer, size_t length, loff_t *offset)
{
    printk("write not offered by %s\n", DEVICE_NAME);
    return -EINVAL;
}

static void pir_timer_register(struct timer_list *ptimer, unsigned long timeover)
{
    init_timer(ptimer);
    ptimer->expires = get_jiffies_64() + timeover;
    ptimer->data = NULL;
    ptimer->function = pir_timer_timeover;
    add_timer(ptimer);
}

static void pir_timer_timeover(unsigned long arg)
{
    printk("PIR motion detection timeout!\n");
    waiting_bit = 0;
    pir_timer_delete(&pir_timer);
    schedule_work(&suspend_work);
}

static void pir_timer_delete(struct timer_list *ptimer)
{
    if (ptimer)
    {
        del_timer(ptimer);
        ptimer = NULL;
    }
}

module_init(init_pir_module);
module_exit(exit_pir_module);

MODULE_LICENSE("GPL");
