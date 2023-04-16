/*
 * oled.h
 */

#ifndef INC_OLED_H_
#define INC_OLED_H_

#include "hbird_sdk_soc.h"


#define I2C_PRESCALER (16000000/(5*100000)) -1 //   i2cfreq = 100Khz
#define I2C_ID        I2C0

#define OLED0561_ADD	0x78  // OLED??I2C????????????
#define COM				0x00  // OLED I2C CMD
#define DAT 			0x40  // OLED I2C DATA


void WriteCmd(unsigned char I2C_Command);//写命
void WriteDat(unsigned char I2C_Data);//写数
void OLED_Init(void);
void OLED_SetPos(unsigned char x, unsigned char y); //设置起始点坐
void OLED_Fill(unsigned char fill_Data);//全屏填充
void OLED_CLS(void);//清屏
void OLED_ON(void);
void OLED_OFF(void);
void OLED_ShowLagData(uint8_t x,uint8_t y,uint32_t num,uint8_t lenth,uint8_t size);
void OLED_ShowStr(unsigned char x, unsigned char y, unsigned char ch[],unsigned char lenth, unsigned char TextSize);
void OLED_ShowCN(unsigned char x, unsigned char y, unsigned char N);
void OLED_DrawBMP(unsigned char x0,unsigned char y0,unsigned char x1,unsigned char y1,unsigned char BMP[]);
void OLED_ShowChar(uint8_t x,uint8_t y,uint32_t chr,uint8_t Char_Size);
unsigned int oled_pow(unsigned int m,unsigned int n);

#endif /* INC_OLED_H_ */
