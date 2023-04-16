#include "weights.h"
#include "insn.h"
DTYPE mode;
DTYPE mulf(DTYPE a, DTYPE b)
{
    if (mode)
        return nice_mulf(a,b);
    else
        return a*b;
}
void sconv0(DTYPE input[513][513][3][1], DTYPE weights[3][3][3][16], DTYPE beta[1][1][1][16], DTYPE output[257][257][16][1])
{
    for (int ih = 0; ih < 512; ih = ih + 2)
        for (int iw = 0; iw < 512; iw = iw + 2)
            for (int ic = 0; ic < 16; ic++)
            {
                DTYPE sum = beta[0][0][0][ic];
                for (int in = 0; in < 3; in++)
                    for (int kh = 0; kh < 3; kh++)
                        for (int kw = 0; kw < 3; kw++)
                            sum += mulf(input[ih + kh][iw + kw][in][0], weights[kh][kw][in][ic]);
                output[ih / 2][iw / 2][ic][0] = (sum > 0 ? sum : 0);
            }
}

void dpconv1(DTYPE input[257][257][16][1], DTYPE dweights[3][3][16][1], DTYPE dbeta[1][1][16][1], DTYPE pweights[1][1][16][16], DTYPE pbeta[1][1][1][16], DTYPE output[129][129][16][1])
{
    for (int ih = 0; ih < 256; ih = ih + 2)
        for (int iw = 0; iw < 256; iw = iw + 2)
        {
            DTYPE outofdconv[16];
            for (int in = 0; in < 16; in++)
            {
                DTYPE sum = dbeta[0][0][in][0];
                for (int kh = 0; kh < 3; kh++)
                    for (int kw = 0; kw < 3; kw++)
                        sum += mulf(input[ih + kh][iw + kw][in][0], dweights[kh][kw][in][0]);
                outofdconv[in] = (sum > 0 ? sum : 0);
            }
            for (int ic = 0; ic < 16; ic++)
            {
                DTYPE sum = pbeta[0][0][0][ic];
                for (int in = 0; in < 16; in++)
                    sum += mulf(outofdconv[in], pweights[0][0][in][ic]);
                output[ih / 2][iw / 2][ic][0] = (sum > 0 ? sum : 0);
            }
        }
}

void dpconv2(DTYPE input[129][129][16][1], DTYPE dweights[3][3][16][1], DTYPE dbeta[1][1][16][1], DTYPE pweights[1][1][16][32], DTYPE pbeta[1][1][1][32], DTYPE output[65][65][32][1])
{
    for (int ih = 0; ih < 128; ih = ih + 2)
        for (int iw = 0; iw < 128; iw = iw + 2)
        {
            DTYPE outofdconv[16];
            for (int in = 0; in < 16; in++)
            {
                DTYPE sum = dbeta[0][0][in][0];
                for (int kh = 0; kh < 3; kh++)
                    for (int kw = 0; kw < 3; kw++)
                        sum += mulf(input[ih + kh][iw + kw][in][0], dweights[kh][kw][in][0]);
                outofdconv[in] = (sum > 0 ? sum : 0);
            }
            for (int ic = 0; ic < 32; ic++)
            {
                DTYPE sum = pbeta[0][0][0][ic];
                for (int in = 0; in < 16; in++)
                    sum += mulf(outofdconv[in], pweights[0][0][in][ic]);
                output[ih / 2][iw / 2][ic][0] = (sum > 0 ? sum : 0);
            }
        }
}

void dpconv3(DTYPE input[65][65][32][1], DTYPE dweights[3][3][32][1], DTYPE dbeta[1][1][32][1], DTYPE pweights[1][1][32][16], DTYPE pbeta[1][1][1][16], DTYPE output[33][33][16][1])
{
    for (int ih = 0; ih < 64; ih = ih + 2)
        for (int iw = 0; iw < 64; iw = iw + 2)
        {
            DTYPE outofdconv[32];
            for (int in = 0; in < 32; in++)
            {
                DTYPE sum = dbeta[0][0][in][0];
                for (int kh = 0; kh < 3; kh++)
                    for (int kw = 0; kw < 3; kw++)
                        sum += mulf(input[ih + kh][iw + kw][in][0], dweights[kh][kw][in][0]);
                outofdconv[in] = (sum > 0 ? sum : 0);
            }
            for (int ic = 0; ic < 16; ic++)
            {
                DTYPE sum = pbeta[0][0][0][ic];
                for (int in = 0; in < 32; in++)
                    sum += mulf(outofdconv[in], pweights[0][0][in][ic]);
                output[ih / 2][iw / 2][ic][0] = (sum > 0 ? sum : 0);
            }
        }
}

void dpconv4(DTYPE input[33][33][16][1], DTYPE dweights[3][3][16][1], DTYPE dbeta[1][1][16][1], DTYPE pweights[1][1][16][16], DTYPE pbeta[1][1][1][16], DTYPE output[17][17][16][1])
{
    for (int ih = 0; ih < 32; ih = ih + 2)
        for (int iw = 0; iw < 32; iw = iw + 2)
        {
            DTYPE outofdconv[16];
            for (int in = 0; in < 16; in++)
            {
                DTYPE sum = dbeta[0][0][in][0];
                for (int kh = 0; kh < 3; kh++)
                    for (int kw = 0; kw < 3; kw++)
                        sum += mulf(input[ih + kh][iw + kw][in][0], dweights[kh][kw][in][0]);
                outofdconv[in] = (sum > 0 ? sum : 0);
            }
            for (int ic = 0; ic < 16; ic++)
            {
                DTYPE sum = pbeta[0][0][0][ic];
                for (int in = 0; in < 16; in++)
                    sum += mulf(outofdconv[in], pweights[0][0][in][ic]);
                output[ih / 2][iw / 2][ic][0] = (sum > 0 ? sum : 0);
            }
        }
}

void dpconv5(DTYPE input[17][17][16][1], DTYPE dweights[3][3][16][1], DTYPE dbeta[1][1][16][1], DTYPE pweights[1][1][16][16], DTYPE pbeta[1][1][1][16], DTYPE output[9][9][16][1])
{
    for (int ih = 0; ih < 16; ih = ih + 2)
        for (int iw = 0; iw < 16; iw = iw + 2)
        {
            DTYPE outofdconv[16];
            for (int in = 0; in < 16; in++)
            {
                DTYPE sum = dbeta[0][0][in][0];
                for (int kh = 0; kh < 3; kh++)
                    for (int kw = 0; kw < 3; kw++)
                        sum += mulf(input[ih + kh][iw + kw][in][0], dweights[kh][kw][in][0]);
                outofdconv[in] = (sum > 0 ? sum : 0);
            }
            for (int ic = 0; ic < 16; ic++)
            {
                DTYPE sum = pbeta[0][0][0][ic];
                for (int in = 0; in < 16; in++)
                    sum += mulf(outofdconv[in], pweights[0][0][in][ic]);
                output[ih / 2][iw / 2][ic][0] = (sum > 0 ? sum : 0);
            }
        }
}

void dpconv6(DTYPE input[9][9][16][1], DTYPE dweights[3][3][16][1], DTYPE dbeta[1][1][16][1], DTYPE pweights[1][1][16][3], DTYPE pbeta[1][1][1][3], DTYPE output[5][5][3][1])
{
    for (int ih = 0; ih < 8; ih = ih + 2)
        for (int iw = 0; iw < 8; iw = iw + 2)
        {
            DTYPE outofdconv[16];
            for (int in = 0; in < 16; in++)
            {
                DTYPE sum = dbeta[0][0][in][0];
                for (int kh = 0; kh < 3; kh++)
                    for (int kw = 0; kw < 3; kw++)
                        sum += mulf(input[ih + kh][iw + kw][in][0], dweights[kh][kw][in][0]);
                outofdconv[in] = (sum > 0 ? sum : 0);
            }
            for (int ic = 0; ic < 3; ic++)
            {
                DTYPE sum = pbeta[0][0][0][ic];
                for (int in = 0; in < 16; in++)
                    sum += mulf(outofdconv[in], pweights[0][0][in][ic]);
                output[ih / 2][iw / 2][ic][0] = (sum > 0 ? sum : 0);
            }
        }
}

void avep7(DTYPE input[5][5][3][1], DTYPE output[3])
{
    for (int in = 0; in < 3; in++)
    {
        DTYPE sum=0;
        for (int ih = 0; ih < 4; ih = ih + 1)
            for (int iw = 0; iw < 4; iw = iw + 1)
                sum += input[ih][iw][in][0];
        output[in] = sum / 4 / 4;
    }
}

void DCNN(DTYPE image[513][513][3][1], 
  DTYPE p1out[257][257][16][1], 
  DTYPE p2out[129][129][16][1], 
  DTYPE p3out[65][65][32][1], 
  DTYPE p4out[33][33][16][1], 
  DTYPE p5out[17][17][16][1], 
  DTYPE p6out[9][9][16][1], 
  DTYPE p7out[5][5][3][1], 
  DTYPE output[3])
{
    sconv0(image, scw_1, scb_1, p1out);
    dpconv1(p1out, dcw_1, dcb_1, scw_2, scb_2, p2out);
    dpconv2(p2out, dcw_2, dcb_2, scw_3, scb_3, p3out);
    dpconv3(p3out, dcw_3, dcb_3, scw_4, scb_4, p4out);
    dpconv4(p4out, dcw_4, dcb_4, scw_5, scb_5, p5out);
    dpconv5(p5out, dcw_5, dcb_5, scw_6, scb_6, p6out);
    dpconv6(p6out, dcw_6, dcb_6, scw_7, scb_7, p7out);
    avep7(p7out, output);
}
