
#include "hbird_sdk_soc.h"
#include "math.h"

#define int32 int
#define int16 short
#define int8 char
#define uint8 unsigned char

// HW defines
#define IMGHW 256
#define HW1 (IMGHW/2)
#define C1 16
#define HW2 (HW1/2)
#define C2 16
#define HW3 (HW2/2)
#define C3 32
#define HW4 (HW3/2)
#define C4 32
#define HW5 (HW4/2)
#define C5 64

#define HWOUT HW5
#define COUT 21
#define ZOOM (IMGHW/HWOUT)
#define FEATHW HWOUT
// boxes defines
#define NUM_ANCHOR 3
#define NUM_CLASS 2 //
#define MAX_BOXNUM 10
#define NMS_IOU 0.1 //
#define CONF_THRESHOLD 0.15 //

#define OH 480
#define OW 640
// min max defines
#define MAX(a,b) ((a>b)? a:b)
#define MIN(a,b) ((a<b)? a:b)

/*
 * sigmoid function
 */
float sigmoid(float x);

/*
 * Sort boxes according to confidence
 * boxes: boxes store sequence
 * box : new boxes to insert
 */
int sort(float boxes[MAX_BOXNUM][NUM_CLASS + 5], float box[NUM_CLASS + 5]);
/*
 * Attention!!! Please initialize @para nmsboxes with zeros
 */
int NMS(float boxes[MAX_BOXNUM][NUM_CLASS + 5], float nmsboxes[MAX_BOXNUM][NUM_CLASS + 5]);
/*
 * yolo_head
 * Boxes are the output of whole model.
 * void * data: output of ai model
 * boxes : provide MAX_BOXNUM boxes (x1,y1, x2,y2, confidence, mark0, mark1)
 */
int yolo_head(float out_data[HWOUT][HWOUT][COUT], float boxes[MAX_BOXNUM][NUM_CLASS + 5]);

/*
* boxes print
*/
void boxprint(float nmsboxes[MAX_BOXNUM][NUM_CLASS + 5]);
