#include "insn.h"
#include "model.h"
#include "oled.h"
#include "eth.h"
#include "video.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "hbird_sdk_soc.h"
#include "hbirdv2.h"
#include "hbirdv2_uart.h"
// 63:32 udp fre=125M/reg  25000000
// 31:24 lor 50-250  23:16 130-240 : 150, 185
// nice 15:4 udp length 128
// 3 udp send  2 dcnn  1 hdmi  0 cam : 3
#define LED4 1<<22u
#define LED5 1<<23u
// key3 4 A20 A21
#define KEY3 1<<20u
#define KEY4 1<<21u
extern unsigned cont1,cont2;

void led_test(void);
void oled_test(void);
void gpio_config(void);

void nice_yuntai(unsigned lor, unsigned uod);

void nice_dcnn_on(void);
void nice_dcnn_off(void);


void read_ddr( int *startp, int range);
void write_ddr( int *startp, int range);

void nice_ram_test(void);
void nice_fpu_test(void);
void ddr_test(void);
void nice_cont_test(void);
void led_test(void);
void oled_test(void);

void ddr_init(int *startp, int byterange, unsigned num);

void DCNN_test(void);
void DCNN_once(void);
uint8_t uart_rece(UART_TypeDef *uart, uint8_t* rece_buffer);
