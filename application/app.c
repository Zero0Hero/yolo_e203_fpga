#include "nice.h"

void DCNN_test()
{
	printf("\n/******** DCNN begin*********/\n");
	delay_1ms(2000);
	//application();
	unsigned long begin_instret,begin_cycle,end_instret,end_cycle;
	static DTYPE mdlout[3];
	mode=1;
	nice_hdmi_off();
	nice_cam_off();
	nice_dcnn_off();
	begin_instret =  __get_rv_instret();
	begin_cycle   =  __get_rv_cycle();
	//delay_1ms(1000);
	DCNN(0xA0F00000, 0xA0E80000, 0xA0E00000, 0xA0D80000, 0xA0D00000, 0xA0C80000,0xA0C00000, 0xA0000000, 0xA0001000);
	end_instret = __get_rv_instret();
	end_cycle   = __get_rv_cycle();
	printf("Software DCNN with nice_fpu : %ld  %ld cycles\n",end_instret-begin_instret,end_cycle-begin_cycle);
	mode=0;
	begin_instret =  __get_rv_instret();
	begin_cycle   =  __get_rv_cycle();
	//delay_1ms(1000);
	DCNN(0xA0F00000, 0xA0E80000, 0xA0E00000, 0xA0D80000, 0xA0D00000, 0xA0C80000,0xA0C00000, 0xA0000000, 0xA0001000);
	end_instret = __get_rv_instret();
	end_cycle   = __get_rv_cycle();
	printf("Software DCNN without nice_fpu : %ld  %ld cycles\n",end_instret-begin_instret,end_cycle-begin_cycle);

	int * addr=0xA0001000;
	*(addr)=1;

	begin_instret =  __get_rv_instret();
	begin_cycle   =  __get_rv_cycle();
	//delay_1ms(1000);
	nice_dcnn_on();
	while(*(addr)==1);
	end_instret = __get_rv_instret();
	end_cycle   = __get_rv_cycle();
	printf("Hardware DCNN : %ld  %ld cycles\n",end_instret-begin_instret,end_cycle-begin_cycle);
	nice_hdmi_on();
	nice_cam_on();
	nice_dcnn_off();
	printf("\n/********* DCNN end**********/\n");
}
uint8_t ylor=150,yuod=200,en_cam_run=0,en_img_eth=0;
void DCNN_once()
{
	printf("DCNN once\n");
	char c[60]={0};
	unsigned temp[3]={0};
	unsigned * addr= 0xA0001000;
	volatile unsigned last;
	last=*(addr);
	volatile unsigned now=0;
	//nice_yuntai(60,140);
	nice_dcnn_on();
	delay_1ms(200);
	//while(last==addr[0]);
	gpio_toggle(GPIOA, LED4);
	unsigned max=0;unsigned maxid=0;
	now=addr[0]; last=now;
	temp[0]=now<<16>>16;max=temp[0];
	temp[1]=now>>16;
	if(temp[1]>max)
	{
		max=temp[1];maxid=1;
	}
	now=addr[1];
	temp[2]=now<<16>>16;
	if(temp[2]>max)
	{
		max=temp[2];maxid=2;
	}

	printf("DCNN : %x  %x  %x  %x \n", (int)(temp[0]),(int)(temp[1]),(int)(temp[2]),(int)(maxid));
	sprintf(c,"%4x %4x %4x %4x ", (int)(temp[0]),(int)(temp[1]),(int)(temp[2]),(int)(maxid));
	OLED_ShowStr(2, 1, c,20, 1);

	unsigned short * addrus= 0xA0000000;
	unsigned short temp4, max4=0,max4x=0,max4y=0,class4;
	int i,j;
	for(i=0;i<4;i++)
	{
		printf("\n");

		for(j=0;j<4;j++)
		{
			temp4=*(addrus+i*15+j*3);
			printf("[%x,", (temp4));
			temp4=*(addrus+i*15+j*3+1);
			if(temp4>max4)
			{
				max4=temp4;
				max4x=i;
				max4y=j;
				class4=1;
			}
			printf("%x,", (temp4));
			temp4=*(addrus+i*15+j*3+2);
			if(temp4>max4)
			{
				max4=temp4;
				max4x=i;
				max4y=j;
				class4=2;
			}
			printf("%x]", (temp4));
		}
	}
	printf("\n%x  %x  %x  %x \n", (int)(max4x),(int)(max4y),(int)(temp4),(int)(class4));
	sprintf(c,"%4x %4x %4x %4x ", (int)(max4x),(int)(max4y),(int)(temp4),(int)(class4));
	OLED_ShowStr(2, 3, c,20, 1);
	if(gpio_read(GPIOA, KEY3)==0)
	{
		en_cam_run=!en_cam_run;
		if(en_cam_run)OLED_ShowStr(2, 5, "run on ",7, 1);
		else OLED_ShowStr(2, 5, "run off",7, 1);
	}
	if(gpio_read(GPIOA, KEY4)==0)
	{
		en_img_eth=!en_img_eth;
		if(en_img_eth)OLED_ShowStr(2, 7, "eth on ",7, 1);
		else OLED_ShowStr(2, 7, "eth off",7, 1);
	}
	if(maxid>0 && en_cam_run)
	{
		if(max4y>1&&ylor>60)ylor-=3;
		else if(max4y<1&&ylor<240)ylor+=3;
		if(max4x<1&&yuod>130)yuod-=3;
		else if(max4x>1&&yuod<180)yuod+=3;

		nice_yuntai(ylor, yuod);
	}
	if(maxid==2 && en_img_eth)
		eth_once();
	gpio_toggle(GPIOA, LED4);
}



