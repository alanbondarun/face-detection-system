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
    int s = 0;
    while (1)
    {
        char buf[256];
        read(volup_fd, buf, 256);

        if (buf[1])
        {
           //write(led4_fd, "1", 2);
            s = system("avconv -f video4linux2 -s 320x240 -r 10 -i /dev/video0 -frames 3 img_%d.bmp");
            if(s==-1){
                printf("error in avconv\n");
            } 
        }
        else
        {
            //write(led4_fd, "0", 2);
            remove("img_3.bmp");
            printf("finish video capture ");
        }
        printf("PIR input = %d, led on = %d\n", buf[0], buf[1]);
        usleep(1000000);
    }

    close(volup_fd);
    close(led4_fd);
}
