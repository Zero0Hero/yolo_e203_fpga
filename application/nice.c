#include "nice.h"
#include "eth.h"

// 63:32 udp fre=125M/reg  25000000
// 31:24 lor 50-250  23:16 130-240 : 150, 185
// nice 15:4 udp length 128
// 3 udp send  2 dcnn  1 hdmi  0 cam : 3
unsigned cont1=0x7db40800,cont2=25000000;
void nice_init()
{
	nice_cont(cont2, cont1);
}
void nice_yuntai(unsigned lor, unsigned uod)
{
// 31:24 lor 50-250  23:16 130-240 : 150, 185
	cont1=cont1&0x0000ffff;
	cont1=cont1|(lor<<24)|(uod<<16);
	nice_cont(cont2, cont1);
}

void nice_dcnn_on()
{
	cont1=cont1&0xfffffffb;
	nice_cont(cont2, cont1);
	delay_1ms(1);
	cont1=cont1|0x00000004;
	nice_cont(cont2, cont1);
}
void nice_dcnn_off()
{
	cont1=cont1&0xfffffffb;
	nice_cont(cont2, cont1);
}


void read_ddr( int *startp, int range)
{
	int *p = startp;
	for(int i=0; i < range ; i++ )
	{
		printf(" %d ",*p);
		delay_1ms(10);
		p=p+1;
	}
}
void write_ddr( int *startp, int range)
{
	int *p = startp;
	for(int i=0; i < range ; i++ )
	{
		*p = i;
		p=p+1;
		printf(".");
		delay_1ms(1);
	}
}
uint8_t uart_rece(UART_TypeDef *uart, uint8_t* rece_buffer)
{

	while(uart_read(UART2)!=0x31);
	//if(rece_buffer[0]=0x31)
	{	rece_buffer[0]=0x31;
		for(int i=1;i<7;i++)
		{
			rece_buffer[i]=uart_read(UART2);
		}
		rece_buffer[5]=0;
		printf("%s\n",rece_buffer);
	}
}
void nice_ram_test()
{
	printf("\n/********start nice ram*********/\n");
	for (int i=0;i<4;i++)
		nice_ram_write(i, -i);
	//nice_ram_write(3, -1);
	for (int i=0;i<4;i++)
	{
		int d=nice_ram_read(i);
		printf("%d, ",d);
	}
	printf("\n/********end nice ram***********/\n");

}

void nice_fpu_test()
{
	int begin_instret =  __get_rv_instret();
	int begin_cycle   =  __get_rv_cycle();

	int end_instret = __get_rv_instret();
	int end_cycle   = __get_rv_cycle();


	printf ("\n/*******System start (fpu)**********/\n");

	float a=2.323,b=3.123,c;
	c=nice_mulf(a,b);
	printf("fpu :%d \n",(int)c);

	begin_instret =  __get_rv_instret();
	begin_cycle   =  __get_rv_cycle();
	c=nice_mulf(-a,b);
	end_instret = __get_rv_instret();
	end_cycle   = __get_rv_cycle();
	printf("fpu :%d \n %d  %d\n",(int)c,end_instret-begin_instret,end_cycle-begin_cycle);

	begin_instret =  __get_rv_instret();
	begin_cycle   =  __get_rv_cycle();
	c=-a*b;
	end_instret = __get_rv_instret();
	end_cycle   = __get_rv_cycle();
	printf("nonfpu :%d  \n %d  %d\n",(int)c,end_instret-begin_instret,end_cycle-begin_cycle);

	printf("\n/************end of fpu**************/\n");

}

void ddr_test()
{
	printf("\n/**********start ddr test *************/\n");
	write_ddr(0xA0002000,10);
	delay_1ms(1000);
	read_ddr(0xA0002000,10);
	printf("\n/********** end ddr test **************/\n");
}

void nice_cont_test()
{
	printf("\n/*******start nice control**********/\n");
	nice_cam_off();
	delay_1ms(100);
	nice_hdmi_off();
	delay_1ms(1000);

	nice_cam_on();
	delay_1ms(100);
	nice_hdmi_on();
	delay_1ms(100);
	printf("\n/*********end nice control**********/\n");
}

void led_test()
{
	printf("\n/******* start led test **********/\n");
	gpio_config();
	gpio_toggle(GPIOA, LED4);
	int i=0;
	while(i<5)
	{
		gpio_toggle(GPIOA, LED4);
		gpio_toggle(GPIOA, LED5);
		delay_1ms(500);
		gpio_toggle(GPIOA, LED4);
		gpio_toggle(GPIOA, LED5);
		delay_1ms(500);
		printf("%ds, ",++i);
	}
	printf("\n/******* end led test **********/\n");
}

void oled_test()
{
	printf("\n/*******start oled test **********/\n");
	OLED_Init();
	OLED_ON();
	OLED_CLS();
	delay_1ms(1000);
	OLED_ShowStr(0,0,"oled test:",7,1);
	int i=0;
	while(i<2)
	{
		delay_1ms(1000);
		OLED_ShowLagData(64,4,i++,6,1);
	}
	printf("\n/******* end oled test **********/\n");
	OLED_CLS();
}

void gpio_config()
{
	gpio_enable_output(GPIOA, LED4);
	gpio_enable_output(GPIOA, LED5);
	// key3 4 A20 A21
	gpio_enable_input(GPIOA, 1<<20);
	gpio_enable_input(GPIOA, 1<<21);
}

void ddr_init(int *startp, int byterange, unsigned num) //half 0
{
	int i=0;
	while(i++<byterange/4)
	{
		*startp=num;
		startp+=1;
	}
}



