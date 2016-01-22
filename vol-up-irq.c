#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/interrupt.h>
#include <linux/sched.h>
#include <linux/fs.h>
#include <linux/workqueue.h>
#include <linux/device.h>
#include <asm/uaccess.h>

#define VOL_UP_IRQ 115
#define DEVICE_NAME "volumeup"

static int dummy_val = 0;

static int device_open(struct inode *, struct file *);
static int device_release(struct inode *, struct file *);
static ssize_t device_read(struct file *, char *, size_t, loff_t *);
static ssize_t device_write(struct file *, const char *, size_t, loff_t *);

static struct class *device_class;
static dev_t devdev;

static int major_num;
static int is_device_open = 0;
static int led_up = 0;

static struct file_operations fops = {
    .read = device_read,
    .write = device_write,
    .open = device_open,
    .release = device_release
};

irqreturn_t irq_handler(int irq, void *dev_id)
{
    static int vol_btn_down = 0;

    vol_btn_down = !vol_btn_down;
    if (vol_btn_down)
    {
        led_up = !led_up;
        if (led_up)
        {
            printk("LED up\n");
        }   
        else
        {
            printk("LED down\n");
        }
    }

    return IRQ_HANDLED;
}

int init_module()
{
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

    int request_ret = request_irq(VOL_UP_IRQ,
            irq_handler,
            IRQF_SHARED | IRQF_TRIGGER_RISING | IRQF_TRIGGER_FALLING,
            "test volume up irq handler",
            &dummy_val);

    if (request_ret != 0)
    {
        printk("ERROR: cannot request IRQ %d, error code %d\n", VOL_UP_IRQ, request_ret);
        return request_ret;
    }

    return 0;
}

void cleanup_module()
{
    free_irq(VOL_UP_IRQ, &dummy_val);
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
