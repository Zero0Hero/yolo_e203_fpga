#include "defines.h"
#include <cmath>
#include "yolo_weights.h"


void HDMI_Frame( unsigned short in[OH][OW], unsigned short out[OH][OH])
{
#pragma HLS INLINE
	toh:for (int i = 0; i < OH; i++)
	{
#pragma HLS PIPELINE off
		tow:for (int j = 0; j < OH; j++)
		{
			out[i][j]= in[i][j+(OW-OH)/2];
		}
	}
}

void Downsampling( DTYPE in[OH][OW][3], DTYPE out[IMGHW+2][IMGHW+2][3])
{
#pragma HLS INLINE
	toh:for (int i = 0; i < IMGHW; i++)
	{
#pragma HLS PIPELINE off
		tow:for (int j = 0; j < IMGHW; j++)
		{
			out[i+1][j+1][0]= in[i*OH/IMGHW+1][j*OH/IMGHW+1+(OW-OH)/2][0];
			out[i+1][j+1][1]= in[i*OH/IMGHW+1][j*OH/IMGHW+1+(OW-OH)/2][1];
			out[i+1][j+1][2]= in[i*OH/IMGHW+1][j*OH/IMGHW+1+(OW-OH)/2][2];
		}
		out[i+1][0][0]=0;
		out[i+1][0][1]=0;
		out[i+1][0][2]=0;
		out[i+1][IMGHW+1][0]=0;
		out[i+1][IMGHW+1][1]=0;
		out[i+1][IMGHW+1][2]=0;
	}
	for (int j = 0; j < IMGHW+2; j++)
	{
#pragma HLS PIPELINE off
		out[0][j][0]= 0;
		out[0][j][1]= 0;
		out[0][j][2]= 0;
		out[IMGHW+1][j][0]= 0;
		out[IMGHW+1][j][1]= 0;
		out[IMGHW+1][j][2]= 0;
	}
}

void RGB565toDTYPE(unsigned short RGB565[OH][OW], DTYPE IMG_P[OH][OW][3])
{
#pragma HLS INLINE
	toh:for (int i = 0; i < OH; i++)
	{
#pragma HLS PIPELINE off
		tow:for (int j = 0; j < OW; j++)
		{
			unsigned short sumr=0;
			unsigned short sumg=0;
			unsigned short sumb=0;
			sumb=(unsigned char)((RGB565[i][j]>>11)<<3)+4;
			sumg=(unsigned char)(((RGB565[i][j]>>5)<<10)>>8)+4;
			sumr=(unsigned char)((RGB565[i][j]<<11)>>8)+4;
			IMG_P[i][j][0]= (sumr)/(DTYPE)255.0;
			IMG_P[i][j][1]= (sumg)/(DTYPE)255.0;
			IMG_P[i][j][2]= (sumb)/(DTYPE)255.0;
		}
	}
}

//step=1
template<int H, int W, int N, int C, int K>
class CONV
{
public:
	//standard_conv();
	void conv(DTYPE input[H + 2][W + 2][N], float output[H][W][C],
		const DTYPE weights[3][3][N][C], const DTYPE bias[C])
	{
#pragma HLS INLINE
		for (int16 ih = 0; ih < H; ih = ih + 1)
		{
#pragma HLS PIPELINE off
			for (int16 iw = 0; iw < W; iw = iw + 1)
			{
#pragma HLS PIPELINE off
				//input buffer
				DTYPE input_buffer[3][3][N] = { 0 };
#pragma HLS ARRAY_PARTITION dim=0 variable=input_buffer
				float sum[C] = { 0 };
#pragma HLS ARRAY_PARTITION dim=0 variable=sum
				DTYPE out_buf[C] = { 0 };
#pragma HLS ARRAY_PARTITION dim=0 variable=out_buf
				LOOP_IN:
				for (int16 in = 0; in < N; in++)
				{
#pragma HLS PIPELINE
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
#pragma HLS PIPELINE
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

				/*//outbuffer
				for (int16 ic = 0; ic < C; ic++)
				{

				}*/
			}
		}

	}
};

template<int H, int W, int N, int C, int S>
class CONV_BN_L
{
public:
	//standard_conv();
	void conv(DTYPE input[H+2][W+2][N], DTYPE output[H/S+2][W/S+2][C],
		const DTYPE weights[3][3][N][C], 
		const DTYPE gamma[C], const DTYPE beta[C], const DTYPE mean[C], const DTYPE var[C])
	{
#pragma HLS INLINE
		for (int16 ih = 0; ih < H; ih = ih + S)
		{
#pragma HLS PIPELINE off
			for (int16 iw = 0; iw < W; iw = iw + S)
			{
#pragma HLS PIPELINE off
//#pragma HLS DATAFLOW
				//input buffer
				DTYPE input_buffer[3][3][N] = { 0 };
#pragma HLS ARRAY_PARTITION dim=0 variable=input_buffer
				float sum[C] = { 0 };
#pragma HLS ARRAY_PARTITION dim=0 variable=sum
				DTYPE out_buf[C] = { 0 };
#pragma HLS ARRAY_PARTITION dim=0 variable=out_buf
				LOOP_IN:
				for (int16 in = 0; in < N; in++)
				{
#pragma HLS PIPELINE
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
#pragma HLS PIPELINE
					DTYPE temp;
					sum[ic] = 0;
					for (int16 in = 0; in < N; in++)
					{
						for (int16 kh = 0; kh < 3; kh++)
						{
							for (int16 kw = 0; kw < 3; kw++)
							{
								sum[ic] += input_buffer[kh][kw][in] * weights[kh][kw][in][ic];
							}
						}
					}
					//Batch Normalization
					temp = (sum[ic] - mean[ic]) / sqrtf(var[ic] + (DTYPE)0.001) * gamma[ic] + beta[ic];
					//LeakyReLU
					out_buf[ic] = (temp > 0) ? temp : (temp * (DTYPE)0.125);
					output[ih/S+1][iw/S+1][ic] = out_buf[ic];
				}

				//outbuffer
				/*LOOP_OUT:
				for (int16 ic = 0; ic < C; ic++)
				{
#pragma HLS PIPELINE OFF

				}*/
			}
		}

	}
};

template<int H, int W, int N, int C, int S>
class DPCONV_BN_L
{
public:
	//depthwise_conv();
	void dpconv(DTYPE input[H+2][W+2][N], DTYPE output[H/S+2][W/S+2][C],
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
				//#pragma HLS DATAFLOW
				//input buffer
				DTYPE input_buffer[3][3][N];
#pragma HLS ARRAY_PARTITION dim=0 variable=input_buffer
				float sumd[N] = { 0 };
#pragma HLS ARRAY_PARTITION dim=0 variable=sumd
				float sump[C] = { 0 };
#pragma HLS ARRAY_PARTITION dim=0 variable=sumd
				DTYPE outofpconv[C] = { 0 };
#pragma HLS ARRAY_PARTITION dim=0 variable=outofpconv
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

					/*//kernel depthconv
					for (int16 kh = 0; kh < 3; kh++)
					{
						for (int16 kw = 0; kw < 3; kw++)
						{

						}
					}*/

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
				/*for (int16 ic = 0; ic < C; ic++)
				{

				}*/
			}
		}

	}
};

void yolo_net(unsigned short RGB565[OH][OW], unsigned short HDMIF[OH][OH],
		DTYPE IMG_P[OH][OW][3], DTYPE modelin[IMGHW + 2][IMGHW + 2][3],
		DTYPE out1[HW1 + 2][HW1 + 2][C1], DTYPE out2[HW1 + 2][HW1 + 2][C1],
		DTYPE out3[HW2 + 2][HW2 + 2][C2], DTYPE out4[HW2 + 2][HW2 + 2][C2],
		DTYPE out5[HW3 + 2][HW3 + 2][C3], DTYPE out6[HW3 + 2][HW3 + 2][C3],
		DTYPE out7[HW4 + 2][HW4 + 2][C4], DTYPE out8[HW4 + 2][HW4 + 2][C4],
		DTYPE out9[HW5 + 2][HW5 + 2][C5], DTYPE out10[HW5 + 2][HW5 + 2][C5],
		float out[HWOUT][HWOUT][COUT])
{
#pragma HLS INTERFACE m_axi port=RGB565 offset=direct bundle=img565
#pragma HLS INTERFACE m_axi port=HDMIF offset=direct bundle=HDMI
#pragma HLS INTERFACE m_axi port=IMG_P offset=direct bundle=imgpost
#pragma HLS INTERFACE m_axi port=modelin offset=direct bundle=modelin
#pragma HLS INTERFACE m_axi port=out1 offset=direct bundle=out1
#pragma HLS INTERFACE m_axi port=out2 offset=direct bundle=out2
#pragma HLS INTERFACE m_axi port=out3 offset=direct bundle=out3
#pragma HLS INTERFACE m_axi port=out4 offset=direct bundle=out4
#pragma HLS INTERFACE m_axi port=out5 offset=direct bundle=out5
#pragma HLS INTERFACE m_axi port=out6 offset=direct bundle=out6
#pragma HLS INTERFACE m_axi port=out7 offset=direct bundle=out7
#pragma HLS INTERFACE m_axi port=out8 offset=direct bundle=out8
#pragma HLS INTERFACE m_axi port=out9 offset=direct bundle=out9
#pragma HLS INTERFACE m_axi port=out10 offset=direct bundle=out10
#pragma HLS INTERFACE m_axi port=out offset=direct bundle=out


	CONV_BN_L<IMGHW, IMGHW, 3, C1, 2> conv1;
	DPCONV_BN_L<HW1, HW1, C1, C1, 1> conv2;
	DPCONV_BN_L<HW1, HW1, C1, C2, 2> conv3;
	DPCONV_BN_L<HW2, HW2, C2, C2, 1> conv4;

	DPCONV_BN_L<HW2, HW2, C2, C3, 2> conv5;
	DPCONV_BN_L<HW3, HW3, C3, C3, 1> conv6;
	DPCONV_BN_L<HW3, HW3, C3, C4, 2> conv7;
	DPCONV_BN_L<HW4, HW4, C4, C4, 1> conv8;

	DPCONV_BN_L<HW4, HW4, C4, C5, 2> conv9;
	DPCONV_BN_L<HW5, HW5, C5, C5, 1> conv10;

	CONV<HWOUT, HWOUT, C5, COUT, 3>conv11;

	HDMI_Frame(RGB565, HDMIF);
	RGB565toDTYPE(RGB565, IMG_P);
	Downsampling(IMG_P, modelin);
	conv1.conv(modelin, out1, w0, w1, w2, w3, w4);
	conv2.dpconv(out1, out2, w5, w6, w7, w8, w9, w10);
	conv3.dpconv(out2, out3, w11, w12, w13, w14, w15, w16);
	conv4.dpconv(out3, out4, w17, w18, w19, w20, w21, w22);
	conv5.dpconv(out4, out5, w23, w24, w25, w26, w27, w28);
	conv6.dpconv(out5, out6, w29, w30, w31, w32, w33, w34);
	conv7.dpconv(out6, out7, w35, w36, w37, w38, w39, w40);
	conv8.dpconv(out7, out8, w41, w42, w43, w44, w45, w46);
	conv9.dpconv(out8, out9, w47, w48, w49, w50, w51, w52);
	conv10.dpconv(out9, out10, w53, w54, w55, w56, w57, w58);
	conv11.conv(out10, out, w59, w60);

	return;
}

void yolo_net_test(	DTYPE modelin[IMGHW + 2][IMGHW + 2][3],
		DTYPE out1[HW1 + 2][HW1 + 2][C1], DTYPE out2[HW1 + 2][HW1 + 2][C1],
		DTYPE out3[HW2 + 2][HW2 + 2][C2], DTYPE out4[HW2 + 2][HW2 + 2][C2],
		DTYPE out5[HW3 + 2][HW3 + 2][C3], DTYPE out6[HW3 + 2][HW3 + 2][C3],
		DTYPE out7[HW4 + 2][HW4 + 2][C4], DTYPE out8[HW4 + 2][HW4 + 2][C4],
		DTYPE out9[HW5 + 2][HW5 + 2][C5], DTYPE out10[HW5 + 2][HW5 + 2][C5],
		float out[HWOUT][HWOUT][COUT])
{
	CONV_BN_L<IMGHW, IMGHW, 3, C1, 2> conv1;
	DPCONV_BN_L<HW1, HW1, C1, C1, 1> conv2;
	DPCONV_BN_L<HW1, HW1, C1, C2, 2> conv3;
	DPCONV_BN_L<HW2, HW2, C2, C2, 1> conv4;

	DPCONV_BN_L<HW2, HW2, C2, C3, 2> conv5;
	DPCONV_BN_L<HW3, HW3, C3, C3, 1> conv6;
	DPCONV_BN_L<HW3, HW3, C3, C4, 2> conv7;
	DPCONV_BN_L<HW4, HW4, C4, C4, 1> conv8;

	DPCONV_BN_L<HW4, HW4, C4, C5, 2> conv9;
	DPCONV_BN_L<HW5, HW5, C5, C5, 1> conv10;

	CONV<HWOUT, HWOUT, C5, COUT, 3>conv11;

	conv1.conv(modelin, out1, w0, w1, w2, w3, w4);
	conv2.dpconv(out1, out2, w5, w6, w7, w8, w9, w10);
	conv3.dpconv(out2, out3, w11, w12, w13, w14, w15, w16);
	conv4.dpconv(out3, out4, w17, w18, w19, w20, w21, w22);
	conv5.dpconv(out4, out5, w23, w24, w25, w26, w27, w28);
	conv6.dpconv(out5, out6, w29, w30, w31, w32, w33, w34);
	conv7.dpconv(out6, out7, w35, w36, w37, w38, w39, w40);
	conv8.dpconv(out7, out8, w41, w42, w43, w44, w45, w46);
	conv9.dpconv(out8, out9, w47, w48, w49, w50, w51, w52);
	conv10.dpconv(out9, out10, w53, w54, w55, w56, w57, w58);
	conv11.conv(out10, out, w59, w60);

	return;
}
