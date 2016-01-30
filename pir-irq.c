#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/interrupt.h>
#include <linux/gpio.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/time.h>
#include <linux/suspend.h>
#include <linux/pm.h>
#include <linux/pm_wakeup.h>
#include <linux/pm_wakeirq.h>
#include <linux/workqueue.h>
#include <asm/uaccess.h>

#define DEVICE_NAME "pir"
#define IR_GPIO_PORT 28

const static struct timespec waiting_time = {
    .tv_sec = 5,
    .tv_nsec = 0
};

static int device_open(struct inode *, struct file *);
static int device_release(struct inode *, struct file *);
static ssize_t device_read(struct file *, char *, size_t, loff_t *);
static ssize_t device_write(struct file *, const char *, size_t, loff_t *);

static int suspend_valid(suspend_state_t state);
static int suspend_enter(suspend_state_t state);

static int device_on_runtime_suspend(struct device *dev);

static struct class *device_class;
static struct device *pir_device;
static dev_t devdev;

static int major_num;
static int is_device_open = 0;
static char event_detected = 0;
static char waiting_bit = 0;

static int pir_int_num;

static struct timespec time_last_zero_bit;

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
    .runtime_suspend = device_on_runtime_suspend
};

static void suspend_func(struct work_struct *dummy)
{
    int ret;
    if ((ret = pm_suspend(PM_SUSPEND_MEM)) < 0)
    {
        printk("pm_suspend failed, ret=%d", ret);
    }
}

static DECLARE_WORK(suspend_work, suspend_func);

static int suspend_valid(suspend_state_t state)
{
    return (state == PM_SUSPEND_MEM);
}

static int device_on_runtime_suspend(struct device *dev)
{
    int ret;

    printk("device_on_runtime_suspend\n");

    ret = enable_irq_wake(pir_int_num);
    if (ret < 0)
    {
        printk("ERROR at enable_irq_wake: error code %d\n", ret);
        return ret;
    }
    dev_pm_enable_wake_irq(pir_device);

    return 0;
}

static int suspend_enter(suspend_state_t state)
{
    return 0;
}

irqreturn_t irq_handler(int irq, void *dev_id)
{
    event_detected = !event_detected;
    if (event_detected)
    {
        waiting_bit = 1;
    }
    if (waiting_bit && !event_detected)
    {
        time_last_zero_bit = current_kernel_time();
    }

    if (!event_detected)
    {
        schedule_work(&suspend_work);
    }

    return IRQ_HANDLED;
}

static int __init init_pir_module(void)
{
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
    int ret = gpio_request(IR_GPIO_PORT, "IR GPIO");
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
    int request_ret = request_irq(pir_int_num,
            irq_handler,
            IRQF_TRIGGER_RISING | IRQF_TRIGGER_FALLING,
            "PIR IRQ handler",
            NULL);

    if (request_ret != 0)
    {
        printk("ERROR: cannot request IRQ %d, error code %d\n", pir_int_num, request_ret);
        return request_ret;
    }

    /* device wakeup setting */
    ret = device_init_wakeup(pir_device, true);
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
    }

    /* suspend opeartions setup */
    suspend_set_ops(&sops);

    return 0;
}

static void __exit exit_pir_module(void)
{
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
        
        if (!event_detected && waiting_bit)
        {
            struct timespec interval_after_zero = timespec_sub(current_kernel_time(), time_last_zero_bit);

            if (interval_after_zero.tv_sec > waiting_time.tv_sec || (interval_after_zero.tv_sec == waiting_time.tv_sec &&
                        interval_after_zero.tv_nsec >= waiting_time.tv_nsec))
            {
                waiting_bit = 0;
            }
        }

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

module_init(init_pir_module);
module_exit(exit_pir_module);

MODULE_LICENSE("GPL");
