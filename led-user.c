#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

#define LED4 "/sys/class/leds/apq8016-sbc\:green\:user3/brightness"

int main()
{
    int volup_fd = open("/dev/pir", O_RDONLY);
    int led4_fd = open(LED4, O_WRONLY);   

    if (volup_fd < 0 || led4_fd < 0)
    {
        printf("error opening file\n");
        return 0;
    }

    while (1)
    {
        char buf[256];
        read(volup_fd, buf, 256);

        if (buf[0])
        {
            write(led4_fd, "1", 2);
        }
        else
        {
            write(led4_fd, "0", 2);
        }
        printf("vol = %d\n", buf[0]);
        usleep(100000);
    }

    close(volup_fd);
    close(led4_fd);
}
