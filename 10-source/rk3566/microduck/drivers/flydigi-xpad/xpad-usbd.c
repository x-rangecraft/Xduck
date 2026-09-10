#define _POSIX_C_SOURCE 200809L

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/input.h>
#include <linux/uinput.h>
#include <linux/usbdevice_fs.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define DRIVER_NAME "xpad-usbd"
#define USB_VENDOR 0x045eU
#define USB_PRODUCT 0x028eU
#define USB_INTERFACE 0U
#define USB_ENDPOINT_IN 0x81U
#define USB_REPORT_MINIMUM 14
#define USB_REPORT_SIZE 64
#define USB_TIMEOUT_MS 1000U
#define RECONNECT_MS 500L

struct pad_state {
    int32_t keys[11];
    int32_t axes[8];
};

static volatile sig_atomic_t running = 1;

static const uint16_t key_codes[] = {
    BTN_A, BTN_B, BTN_X, BTN_Y, BTN_TL, BTN_TR,
    BTN_SELECT, BTN_START, BTN_MODE, BTN_THUMBL, BTN_THUMBR,
};

static const uint16_t axis_codes[] = {
    ABS_X, ABS_Y, ABS_RX, ABS_RY, ABS_Z, ABS_RZ, ABS_HAT0X, ABS_HAT0Y,
};

static void stop_running(int signo)
{
    (void)signo;
    running = 0;
}

static void sleep_ms(long milliseconds)
{
    const struct timespec request = {
        .tv_sec = milliseconds / 1000L,
        .tv_nsec = (milliseconds % 1000L) * 1000000L,
    };
    (void)nanosleep(&request, NULL);
}

static int read_hex_file(const char *path, unsigned int *value)
{
    FILE *file = fopen(path, "r");
    int result;

    if (file == NULL) {
        return -1;
    }
    result = fscanf(file, "%x", value);
    (void)fclose(file);
    return result == 1 ? 0 : -1;
}

static int read_uint_file(const char *path, unsigned int *value)
{
    FILE *file = fopen(path, "r");
    int result;

    if (file == NULL) {
        return -1;
    }
    result = fscanf(file, "%u", value);
    (void)fclose(file);
    return result == 1 ? 0 : -1;
}

static int find_usb_node(char *node, size_t node_size)
{
    static const char sysfs_root[] = "/sys/bus/usb/devices";
    DIR *directory = opendir(sysfs_root);
    struct dirent *entry;

    if (directory == NULL) {
        return -1;
    }

    while ((entry = readdir(directory)) != NULL) {
        char path[512];
        unsigned int vendor;
        unsigned int product;
        unsigned int bus;
        unsigned int device;

        if (strchr(entry->d_name, ':') != NULL) {
            continue;
        }
        if (snprintf(path, sizeof(path), "%s/%s/idVendor", sysfs_root,
                     entry->d_name) >= (int)sizeof(path) ||
            read_hex_file(path, &vendor) != 0 || vendor != USB_VENDOR) {
            continue;
        }
        if (snprintf(path, sizeof(path), "%s/%s/idProduct", sysfs_root,
                     entry->d_name) >= (int)sizeof(path) ||
            read_hex_file(path, &product) != 0 || product != USB_PRODUCT) {
            continue;
        }
        if (snprintf(path, sizeof(path), "%s/%s/busnum", sysfs_root,
                     entry->d_name) >= (int)sizeof(path) ||
            read_uint_file(path, &bus) != 0) {
            continue;
        }
        if (snprintf(path, sizeof(path), "%s/%s/devnum", sysfs_root,
                     entry->d_name) >= (int)sizeof(path) ||
            read_uint_file(path, &device) != 0) {
            continue;
        }
        if (snprintf(node, node_size, "/dev/bus/usb/%03u/%03u", bus, device) >=
            (int)node_size) {
            continue;
        }
        (void)closedir(directory);
        return 0;
    }

    (void)closedir(directory);
    errno = ENODEV;
    return -1;
}

static int16_t little_endian_i16(const uint8_t *bytes)
{
    uint16_t value = (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8U);
    return (int16_t)value;
}

static int32_t active(uint8_t value, uint8_t mask)
{
    return (value & mask) != 0U ? 1 : 0;
}

static int decode_report(const uint8_t *report, int length, struct pad_state *state)
{
    uint8_t low;
    uint8_t high;

    if (length < USB_REPORT_MINIMUM || report[0] != 0x00U || report[1] < 0x14U) {
        return -1;
    }

    low = report[2];
    high = report[3];
    state->keys[0] = active(high, 0x10U); /* A */
    state->keys[1] = active(high, 0x20U); /* B */
    state->keys[2] = active(high, 0x40U); /* X */
    state->keys[3] = active(high, 0x80U); /* Y */
    state->keys[4] = active(high, 0x01U); /* LB */
    state->keys[5] = active(high, 0x02U); /* RB */
    state->keys[6] = active(low, 0x20U);  /* Back */
    state->keys[7] = active(low, 0x10U);  /* Start */
    state->keys[8] = active(high, 0x04U); /* Guide */
    state->keys[9] = active(low, 0x40U);  /* Left stick */
    state->keys[10] = active(low, 0x80U); /* Right stick */

    state->axes[0] = little_endian_i16(&report[6]);
    state->axes[1] = (int32_t)(int16_t)~(uint16_t)little_endian_i16(&report[8]);
    state->axes[2] = little_endian_i16(&report[10]);
    state->axes[3] = (int32_t)(int16_t)~(uint16_t)little_endian_i16(&report[12]);
    state->axes[4] = report[4];
    state->axes[5] = report[5];
    state->axes[6] = active(low, 0x08U) - active(low, 0x04U);
    state->axes[7] = active(low, 0x02U) - active(low, 0x01U);
    return 0;
}

static int configure_axis(int fd, uint16_t code, int32_t minimum, int32_t maximum,
                          int32_t fuzz, int32_t flat)
{
    struct uinput_abs_setup setup;

    memset(&setup, 0, sizeof(setup));
    setup.code = code;
    setup.absinfo.minimum = minimum;
    setup.absinfo.maximum = maximum;
    setup.absinfo.fuzz = fuzz;
    setup.absinfo.flat = flat;
    return ioctl(fd, UI_ABS_SETUP, &setup);
}

static int create_uinput(void)
{
    struct uinput_setup setup;
    int fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK | O_CLOEXEC);
    size_t index;

    if (fd < 0) {
        return -1;
    }
    if (ioctl(fd, UI_SET_EVBIT, EV_KEY) < 0 ||
        ioctl(fd, UI_SET_EVBIT, EV_ABS) < 0 ||
        ioctl(fd, UI_SET_EVBIT, EV_SYN) < 0) {
        goto fail;
    }
    for (index = 0; index < sizeof(key_codes) / sizeof(key_codes[0]); ++index) {
        if (ioctl(fd, UI_SET_KEYBIT, key_codes[index]) < 0) {
            goto fail;
        }
    }
    for (index = 0; index < sizeof(axis_codes) / sizeof(axis_codes[0]); ++index) {
        if (ioctl(fd, UI_SET_ABSBIT, axis_codes[index]) < 0) {
            goto fail;
        }
    }
    if (configure_axis(fd, ABS_X, -32768, 32767, 16, 128) < 0 ||
        configure_axis(fd, ABS_Y, -32768, 32767, 16, 128) < 0 ||
        configure_axis(fd, ABS_RX, -32768, 32767, 16, 128) < 0 ||
        configure_axis(fd, ABS_RY, -32768, 32767, 16, 128) < 0 ||
        configure_axis(fd, ABS_Z, 0, 255, 0, 0) < 0 ||
        configure_axis(fd, ABS_RZ, 0, 255, 0, 0) < 0 ||
        configure_axis(fd, ABS_HAT0X, -1, 1, 0, 0) < 0 ||
        configure_axis(fd, ABS_HAT0Y, -1, 1, 0, 0) < 0) {
        goto fail;
    }

    memset(&setup, 0, sizeof(setup));
    (void)snprintf(setup.name, sizeof(setup.name), "Flydigi Dune Fox (xpad-usbd)");
    setup.id.bustype = BUS_USB;
    setup.id.vendor = USB_VENDOR;
    setup.id.product = USB_PRODUCT;
    setup.id.version = 0x0110;
    if (ioctl(fd, UI_DEV_SETUP, &setup) < 0 || ioctl(fd, UI_DEV_CREATE) < 0) {
        goto fail;
    }
    return fd;

fail:
    {
        int saved_errno = errno;
        (void)close(fd);
        errno = saved_errno;
    }
    return -1;
}

static int emit_event(int fd, uint16_t type, uint16_t code, int32_t value)
{
    struct input_event event;

    memset(&event, 0, sizeof(event));
    (void)gettimeofday(&event.time, NULL);
    event.type = type;
    event.code = code;
    event.value = value;
    return write(fd, &event, sizeof(event)) == (ssize_t)sizeof(event) ? 0 : -1;
}

static int emit_state(int fd, const struct pad_state *state,
                      const struct pad_state *previous, bool first)
{
    size_t index;
    bool changed = first;

    for (index = 0; index < sizeof(key_codes) / sizeof(key_codes[0]); ++index) {
        if (first || state->keys[index] != previous->keys[index]) {
            if (emit_event(fd, EV_KEY, key_codes[index], state->keys[index]) != 0) {
                return -1;
            }
            changed = true;
        }
    }
    for (index = 0; index < sizeof(axis_codes) / sizeof(axis_codes[0]); ++index) {
        if (first || state->axes[index] != previous->axes[index]) {
            if (emit_event(fd, EV_ABS, axis_codes[index], state->axes[index]) != 0) {
                return -1;
            }
            changed = true;
        }
    }
    return !changed || emit_event(fd, EV_SYN, SYN_REPORT, 0) == 0 ? 0 : -1;
}

static int open_controller(char *node, size_t node_size)
{
    unsigned int interface = USB_INTERFACE;
    int fd;

    if (find_usb_node(node, node_size) != 0) {
        return -1;
    }
    fd = open(node, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        return -1;
    }
    if (ioctl(fd, USBDEVFS_CLAIMINTERFACE, &interface) < 0) {
        int saved_errno = errno;
        (void)close(fd);
        errno = saved_errno;
        return -1;
    }
    return fd;
}

static int read_report(int fd, uint8_t *report)
{
    struct usbdevfs_bulktransfer transfer = {
        .ep = USB_ENDPOINT_IN,
        .len = USB_REPORT_SIZE,
        .timeout = USB_TIMEOUT_MS,
        .data = report,
    };
    return ioctl(fd, USBDEVFS_BULK, &transfer);
}

static int run_self_test(void)
{
    const uint8_t a_pressed[20] = {
        0x00, 0x14, 0x00, 0x10, 0x00, 0x00, 0x34, 0x12, 0x00, 0x80,
        0xcc, 0xed, 0xff, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };
    struct pad_state state;

    memset(&state, 0, sizeof(state));
    if (decode_report(a_pressed, (int)sizeof(a_pressed), &state) != 0 ||
        state.keys[0] != 1 || state.keys[1] != 0 ||
        state.axes[0] != 0x1234 || state.axes[1] != 32767 ||
        state.axes[2] != -0x1234 || state.axes[3] != -32768) {
        fprintf(stderr, DRIVER_NAME ": parser self-test failed\n");
        return EXIT_FAILURE;
    }
    puts(DRIVER_NAME ": parser self-test passed");
    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
    bool absence_logged = false;

    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        return run_self_test();
    }
    if (argc != 1) {
        fprintf(stderr, "usage: %s [--self-test]\n", argv[0]);
        return EXIT_FAILURE;
    }
    (void)signal(SIGINT, stop_running);
    (void)signal(SIGTERM, stop_running);

    while (running != 0) {
        char node[64];
        int usb_fd = open_controller(node, sizeof(node));
        int input_fd;
        struct pad_state previous;
        bool first = true;

        if (usb_fd < 0) {
            if (!absence_logged) {
                fprintf(stderr, DRIVER_NAME ": waiting for %04x:%04x (%s)\n",
                        USB_VENDOR, USB_PRODUCT, strerror(errno));
                absence_logged = true;
            }
            sleep_ms(RECONNECT_MS);
            continue;
        }
        input_fd = create_uinput();
        if (input_fd < 0) {
            fprintf(stderr, DRIVER_NAME ": cannot create /dev/uinput device: %s\n",
                    strerror(errno));
            (void)close(usb_fd);
            sleep_ms(RECONNECT_MS);
            continue;
        }

        absence_logged = false;
        memset(&previous, 0, sizeof(previous));
        fprintf(stderr, DRIVER_NAME ": connected %s as standard evdev gamepad\n", node);

        while (running != 0) {
            uint8_t report[USB_REPORT_SIZE];
            struct pad_state state;
            int length = read_report(usb_fd, report);

            if (length < 0) {
                if (errno == ETIMEDOUT || errno == EINTR) {
                    continue;
                }
                fprintf(stderr, DRIVER_NAME ": USB disconnected: %s\n", strerror(errno));
                break;
            }
            if (decode_report(report, length, &state) != 0) {
                continue;
            }
            if (emit_state(input_fd, &state, &previous, first) != 0) {
                fprintf(stderr, DRIVER_NAME ": evdev write failed: %s\n", strerror(errno));
                break;
            }
            previous = state;
            first = false;
        }

        (void)ioctl(input_fd, UI_DEV_DESTROY);
        (void)close(input_fd);
        (void)close(usb_fd);
        if (running != 0) {
            sleep_ms(RECONNECT_MS);
        }
    }

    return EXIT_SUCCESS;
}
