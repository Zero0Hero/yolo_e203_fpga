/*
 * uart.c
 *
 *  Created on: 2022Äê7ÔÂ17ÈÕ
 *      Author: 17126
 */
#include "uart.h"

void uart_user_init(UART_TypeDef *uart)
{
	uart_init(uart,115200);
	uart_config_stopbit(uart,UART_STOP_BIT_1);
	uart_disable_paritybit(uart);
	uart_enable_rx_th_int(uart);
	uart_set_rx_th(uart,2);
}
uint8_t uart_user_read(UART_TypeDef *uart, uint8_t *buf, uint16_t len)
{
	uint8_t lsr;
	uint16_t index=0;
    if (__RARELY(uart == NULL)) {
        return -1;
    }
    if (len==0)
		do {
			buf[index] = ((uart->RBR)& 0xFF);
			lsr=(uart->LSR & 0x1);
			if(lsr != 0x1)
			{
				delay_1ms(1);
				lsr=(uart->LSR & 0x1);
			}
			index++;
		}
		while (lsr == 0x1);
    else
    	for(index=0;index<len;index++)
    	{
    		while (lsr == 0x0);
			buf[index] = ((uart->RBR)& 0xFF);
		}

    return 0;
}

