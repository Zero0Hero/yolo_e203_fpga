#include "video.h"
#include "nice.h"

void nice_cam_on()
{
	cont1=cont1|0x00000001;
	nice_cont(cont2, cont1);
}
void nice_cam_off()
{
	cont1=cont1&0xfffffffe;
	nice_cont(cont2, cont1);
}
void nice_hdmi_on()
{
	cont1=cont1|0x00000002;
	nice_cont(cont2, cont1);
}
void nice_hdmi_off()
{
	cont1=cont1&0xfffffffd;
	nice_cont(cont2, cont1);
}
