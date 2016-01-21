/* source from www.tldp.org/LDP/lkmpg/2.6/html/x569.html */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <asm/uaccess.h>

int init_module(void);
void cleanup_module(void);
static int device_open(struct inode *, struct file *);
static int device_release(struct inode *, struct file *);
static ssize_t device_read(struct file *, char *, size_t, loff_t *);
static ssize_t device_write(struct file *, const char *, size_t, loff_t *);

static const int SUCCESS = 0;
static const char[] DEVICE_NAME = "chardev";
static const int BUF_LEN = 80;

static int major_num;
static int device_open = 0;

static char msg[BUF_LEN];
static char *msg_ptr;

static struct file_operations fops = {
    .read = device_read,
    .write = device_write,
    .open = device_open,
    .release = device_release
};

int init_module(void)
{
    major_num = register_chrdev(0, DEVICE_NAME, &fops);
    if (major_num < 0)
    {
        printk(KERN_ALERT "registering chardev device failed with %d\n", major_num);
        return major_num;
    }

    printk(KERN_INFO "CHARDEV, major_num = %d\n", major_num);
    printk(KERN_INFO "mknod = /dev/%s c %d 0", DEVICE_NAME, major_num);

    return SUCCESS;
}


