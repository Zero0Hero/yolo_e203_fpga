

// 卷积优化文件
/*********************************************************************/
#include "defines.h"
#include <cmath>
#include "yolo_weights.h"

#define H 256
#define W 256
#define N 16
#define C 16
#define S 2

//dpconv_f
void dpconv_f(DTYPE input[H+2][W+2][N], DTYPE output[H/S+2][W/S+2][C],
	const DTYPE dweights[3][3][N][1], const DTYPE pweights[1][1][N][C],
	const DTYPE gamma[C], const DTYPE beta[C], const DTYPE mean[C], const DTYPE var[C])
{
#pragma HLS INLINE
	for (int16 ih = 0; ih < H; ih = ih + S)
	{
#pragma HLS PIPELINE off
		for (int16 iw = 0; iw < W; iw = iw + S)
		{
#pragma HLS PIPELINE off
			//input buffer
			DTYPE input_buffer[3][3][N];
			float sumd[N] = { 0 };
			float sump[C] = { 0 };
			DTYPE outofpconv[C] = { 0 };
			LOOP_CONV:
			for (int16 in = 0; in < N; in++)
			{
#pragma HLS PIPELINE II=1
				sumd[in] = 0;
				for (int16 kh = 0; kh < 3; kh++)
				{
					for (int16 kw = 0; kw < 3; kw++)
					{
						input_buffer[kh][kw][in] = input[ih + kh][iw + kw][in];
						sumd[in] += input_buffer[kh][kw][in] * dweights[kh][kw][in][0];
					}
				}

				//kernel pointconv
				for (int16 ic = 0; ic < C; ic++)
				{
					sump[ic] += sumd[in] * pweights[0][0][in][ic];
				}
			}
			//outbuffer
			LOOP_OUT:
			for (int16 ic = 0; ic < C; ic++)
			{
#pragma HLS PIPELINE
				DTYPE temp;
				//Batch Normalization
				temp = (sump[ic] - mean[ic]) / sqrtf(var[ic] + (DTYPE)0.001) * gamma[ic] + beta[ic];
				//LeakyReLU
				outofpconv[ic] = (temp > 0) ? temp : (temp * (DTYPE)0.125);
				output[ih/S+1][iw/S+1][ic] = outofpconv[ic];
			}
		}
	}
}


//dpconv_o
void dpconv_o(DTYPE input[H+2][W+2][N], DTYPE output[H/S+2][W/S+2][C],
	const DTYPE dweights[3][3][N][1], const DTYPE pweights[1][1][N][C],
	const DTYPE gamma[C], const DTYPE beta[C], const DTYPE mean[C], const DTYPE var[C])
{
//#pragma HLS INLINE
	for (int16 ih = 0; ih < H; ih = ih + S)
	{
//#pragma HLS PIPELINE off
		for (int16 iw = 0; iw < W; iw = iw + S)
		{
//#pragma HLS PIPELINE off
			//#pragma HLS DATAFLOW
			//input buffer
			DTYPE input_buffer[3][3][N];
//#pragma HLS ARRAY_PARTITION dim=0 variable=input_buffer
			float sumd[N] = { 0 };
//#pragma HLS ARRAY_PARTITION dim=0 variable=sumd
			float sump[C] = { 0 };
//#pragma HLS ARRAY_PARTITION dim=0 variable=sumd
			DTYPE outofpconv[C] = { 0 };
//#pragma HLS ARRAY_PARTITION dim=0 variable=outofpconv
			LOOP_CONV:
			for (int16 in = 0; in < N; in++)
			{
//#pragma HLS PIPELINE II=1
				sumd[in] = 0;
				for (int16 kh = 0; kh < 3; kh++)
				{
					for (int16 kw = 0; kw < 3; kw++)
					{
						input_buffer[kh][kw][in] = input[ih + kh][iw + kw][in];
						sumd[in] += input_buffer[kh][kw][in] * dweights[kh][kw][in][0];
					}
				}

				//kernel pointconv
				for (int16 ic = 0; ic < C; ic++)
				{
					sump[ic] += sumd[in] * pweights[0][0][in][ic];
				}
			}

			//outbuffer
			LOOP_OUT:
			for (int16 ic = 0; ic < C; ic++)
			{
//#pragma HLS PIPELINE
				DTYPE temp;
				//Batch Normalization
				temp = (sump[ic] - mean[ic]) / sqrtf(var[ic] + (DTYPE)0.001) * gamma[ic] + beta[ic];
				//LeakyReLU
				outofpconv[ic] = (temp > 0) ? temp : (temp * (DTYPE)0.125);
				output[ih/S+1][iw/S+1][ic] = outofpconv[ic];
			}
		}
	}

}



void conv(DTYPE input[H + 2][W + 2][N], float output[H][W][C],
		const DTYPE weights[3][3][N][C], const DTYPE bias[C])
{
//#pragma HLS INLINE
	for (int16 ih = 0; ih < H; ih = ih + 1)
	{
//#pragma HLS PIPELINE off
		for (int16 iw = 0; iw < W; iw = iw + 1)
		{
//#pragma HLS PIPELINE off
			//input buffer
			DTYPE input_buffer[3][3][N] = { 0 };
//#pragma HLS ARRAY_PARTITION dim=0 variable=input_buffer
			float sum[C] = { 0 };
//#pragma HLS ARRAY_PARTITION dim=0 variable=sum
			DTYPE out_buf[C] = { 0 };
//#pragma HLS ARRAY_PARTITION dim=0 variable=out_buf
			LOOP_IN:
			for (int16 in = 0; in < N; in++)
			{
//#pragma HLS PIPELINE
				for (int16 kh = 0; kh < 3; kh++)
				{
					for (int16 kw = 0; kw < 3; kw++)
					{
						input_buffer[kh][kw][in] = input[ih + kh][iw + kw][in];
					}
				}
			}

			//kernel
			LOOP_CONV:
			for (int16 ic = 0; ic < C; ic++)
			{

				//DTYPE temp;
				sum[ic] = bias[ic];
				for (int16 in = 0; in < N; in++)
				{
//#pragma HLS PIPELINE
					for (int16 kh = 0; kh < 3; kh++)
					{
						for (int16 kw = 0; kw < 3; kw++)
						{
							sum[ic] += input_buffer[kh][kw][in] * weights[kh][kw][in][ic];
						}
					}
				}
				output[ih][iw][ic] = sum[ic];
			}
		}
	}

}
