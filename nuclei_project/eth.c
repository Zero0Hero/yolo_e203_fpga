#include "eth.h"
#include "nice.h"

void nice_eth_on()
{
	cont1=cont1|0x00000008;
	nice_cont(cont2, cont1);
}
void nice_eth_off()
{
	cont1=cont1&0xfffffff7;
	nice_cont(cont2, cont1);
}
void nice_eth_udp_len(unsigned len, unsigned fre)
{
	cont1=cont1&0xffff000f;
	cont1=cont1|(len<<4);
	cont2=125000000/fre;
	nice_cont(cont2, cont1);
}

void eth_test()
{
	ddr_init(0xA0F7FC00,10*4,0x00ff00ff);
	read_ddr(0xA0F7FC00,10);
	nice_eth_udp_len(128,512);

	nice_eth_on();
	delay_1ms(1000);
	nice_eth_off();
	delay_1ms(1000);
	nice_eth_on();
	delay_1ms(1000);
	nice_eth_off();
	delay_1ms(1000);
	nice_eth_on();
	delay_1ms(1000);
	nice_eth_off();
}

void eth_once()
{
	nice_eth_on();
	delay_1ms(1200);
	nice_eth_off();
	//delay_1ms(500);
}
