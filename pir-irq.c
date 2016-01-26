#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/interrupt.h>
#include <linux/sched.h>
#include <linux/gpio.h>
#include <linux/fs.h>
#include <linux/workqueue.h>
#include <linux/device.h>
#include <asm/uaccess.h>

#define IR_GPIO_PORT 28
#define DEVICE_NAME "pir"

static int device_open(struct inode *, struct file *);
static int device_release(struct inode *, struct file *);
static ssize_t device_read(struct file *, char *, size_t, loff_t *);
static ssize_t device_write(struct file *, const char *, size_t, loff_t *);

static struct class *device_class;
static dev_t devdev;

static int major_num;
static int is_device_open = 0;
static int led_up = 0;

static int pir_int_num;

static struct file_operations fops = {
    .read = device_read,
    .write = device_write,
    .open = device_open,
    .release = device_release
};

irqreturn_t irq_handler(int irq, void *dev_id)
{
    static int event_detected = 0;

    led_up = gpio_get_value(IR_GPIO_PORT);

    return IRQ_HANDLED;
}

int init_module()
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

    devdev = MKDEV(major_num, 0);
    device_create(device_class, NULL, devdev, NULL, DEVICE_NAME);

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

    return 0;
}

void cleanup_module()
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
    int bytes_to_read = 1;

    while (length && bytes_to_read)
    {
        put_user(led_up, buffer++);
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

MODULE_LICENSE("GPL");
