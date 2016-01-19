#include <linux/module.h>

static int __init hellomod_init(void)
{
    printk("Hello world 1\n");
    return 0;
}

static void __exit hellomod_exit(void)
{
    printk("Goodbye world 1\n");
}

module_init(hellomod_init);
module_exit(hellomod_exit);
