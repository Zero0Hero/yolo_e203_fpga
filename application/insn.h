#ifndef __INSN_H__
#define __INSN_H__

#include <hbird_sdk_soc.h>

#define  ROW_LEN    3
#define  COL_LEN    3
#define _DEBUG_INFO_
////////////////////////////////////////////////////////////
// custom3:
// Supported format: only R type here
// Supported instr:
//  1. custom3 lbuf: load data(in memory) to row_buf
//     lbuf (a1)
//     .insn r opcode, func3, func7, rd, rs1, rs2    
//  2. custom3 sbuf: store data(in row_buf) to memory
//     sbuf (a1)
//     .insn r opcode, func3, func7, rd, rs1, rs2    
//  3. custom3 acc rowsum: load data from memory(@a1), accumulate row datas and write back 
//     rowsum rd, a1, x0
//     .insn r opcode, func3, func7, rd, rs1, rs2    
////////////////////////////////////////////////////////////

__STATIC_FORCEINLINE void nice_cont(unsigned cmd_rs2, unsigned cmd_rs1)
{
    int zero = 0;
    asm volatile (
       ".insn r 0x7b, 3, 7, x0, %1, %2"
             :"=r"(zero)
             :"r"(cmd_rs1),"r"(cmd_rs2)
     );
}

// custom read
__STATIC_FORCEINLINE int nice_ram_read(int addr)
{
    int dread;
    int zero = 0;
    asm volatile (
       ".insn r 0x7b, 6, 5, %0, %1, x0"
             :"=r"(dread)
             :"r"(addr)
     );

    return dread;
}
// custom write
__STATIC_FORCEINLINE void nice_ram_write(int addr, int dwrite)
{
    int zero = 0;

    asm volatile (
       ".insn r 0x7b, 3, 4, x0, %1, %2"
           :"=r"(zero)
           :"r"(addr),"r"(dwrite)
     );
}

__STATIC_FORCEINLINE float nice_mulf(float a, float b)
{
    float dread;
    int zero = 0;
    asm volatile (
       ".insn r 0x7b, 7, 6, %0, %1, %2"
             :"=r"(dread)
             :"r"(a),"r"(b)
     );

    return dread;
}

#endif

