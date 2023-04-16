/*
 * uart.h
 *
 *  Created on: 2022Äê7ÔÂ17ÈÕ
 *      Author: 17126
 */

#ifndef APPLICATION_UART_H_
#define APPLICATION_UART_H_
#include "hbird_sdk_soc.h"
void uart_user_init();
uint8_t uart_user_read(UART_TypeDef *uart, uint8_t *buf, uint16_t len);



#endif /* APPLICATION_UART_H_ */
