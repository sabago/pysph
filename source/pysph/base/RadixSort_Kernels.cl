/* ============================================================

Copyright (c) 2009-2010 Advanced Micro Devices, Inc.  All rights reserved.
 
Redistribution and use of this material is permitted under the following 
conditions:
 
Redistributions must retain the above copyright notice and all terms of this 
license.
 
In no event shall anyone redistributing or accessing or using this material 
commence or participate in any arbitration or legal action relating to this 
material against Advanced Micro Devices, Inc. or any copyright holders or 
contributors. The foregoing shall survive any expiration or termination of 
this license or any agreement or access or use related to this material. 

ANY BREACH OF ANY TERM OF THIS LICENSE SHALL RESULT IN THE IMMEDIATE REVOCATION 
OF ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE THIS MATERIAL.

THIS MATERIAL IS PROVIDED BY ADVANCED MICRO DEVICES, INC. AND ANY COPYRIGHT 
HOLDERS AND CONTRIBUTORS "AS IS" IN ITS CURRENT CONDITION AND WITHOUT ANY 
REPRESENTATIONS, GUARANTEE, OR WARRANTY OF ANY KIND OR IN ANY WAY RELATED TO 
SUPPORT, INDEMNITY, ERROR FREE OR UNINTERRUPTED OPERA TION, OR THAT IT IS FREE 
FROM DEFECTS OR VIRUSES.  ALL OBLIGATIONS ARE HEREBY DISCLAIMED - WHETHER 
EXPRESS, IMPLIED, OR STATUTORY - INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED 
WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, 
ACCURACY, COMPLETENESS, OPERABILITY, QUALITY OF SERVICE, OR NON-INFRINGEMENT. 
IN NO EVENT SHALL ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR 
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, PUNITIVE,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, REVENUE, DATA, OR PROFITS; OR 
BUSINESS INTERRUPTION) HOWEVER CAUSED OR BASED ON ANY THEORY OF LIABILITY 
ARISING IN ANY WAY RELATED TO THIS MATERIAL, EVEN IF ADVISED OF THE POSSIBILITY 
OF SUCH DAMAGE. THE ENTIRE AND AGGREGATE LIABILITY OF ADVANCED MICRO DEVICES, 
INC. AND ANY COPYRIGHT HOLDERS AND CONTRIBUTORS SHALL NOT EXCEED TEN DOLLARS 
(US $10.00). ANYONE REDISTRIBUTING OR ACCESSING OR USING THIS MATERIAL ACCEPTS 
THIS ALLOCATION OF RISK AND AGREES TO RELEASE ADVANCED MICRO DEVICES, INC. AND 
ANY COPYRIGHT HOLDERS AND CONTRIBUTORS FROM ANY AND ALL LIABILITIES, 
OBLIGATIONS, CLAIMS, OR DEMANDS IN EXCESS OF TEN DOLLARS (US $10.00). THE 
FOREGOING ARE ESSENTIAL TERMS OF THIS LICENSE AND, IF ANY OF THESE TERMS ARE 
CONSTRUED AS UNENFORCEABLE, FAIL IN ESSENTIAL PURPOSE, OR BECOME VOID OR 
DETRIMENTAL TO ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR 
CONTRIBUTORS FOR ANY REASON, THEN ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE 
THIS MATERIAL SHALL TERMINATE IMMEDIATELY. MOREOVER, THE FOREGOING SHALL 
SURVIVE ANY EXPIRATION OR TERMINATION OF THIS LICENSE OR ANY AGREEMENT OR 
ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE IS HEREBY PROVIDED, AND BY REDISTRIBUTING OR ACCESSING OR USING THIS 
MATERIAL SUCH NOTICE IS ACKNOWLEDGED, THAT THIS MATERIAL MAY BE SUBJECT TO 
RESTRICTIONS UNDER THE LAWS AND REGULATIONS OF THE UNITED STATES OR OTHER 
COUNTRIES, WHICH INCLUDE BUT ARE NOT LIMITED TO, U.S. EXPORT CONTROL LAWS SUCH 
AS THE EXPORT ADMINISTRATION REGULATIONS AND NATIONAL SECURITY CONTROLS AS 
DEFINED THEREUNDER, AS WELL AS STATE DEPARTMENT CONTROLS UNDER THE U.S. 
MUNITIONS LIST. THIS MATERIAL MAY NOT BE USED, RELEASED, TRANSFERRED, IMPORTED,
EXPORTED AND/OR RE-EXPORTED IN ANY MANNER PROHIBITED UNDER ANY APPLICABLE LAWS, 
INCLUDING U.S. EXPORT CONTROL LAWS REGARDING SPECIFICALLY DESIGNATED PERSONS, 
COUNTRIES AND NATIONALS OF COUNTRIES SUBJECT TO NATIONAL SECURITY CONTROLS. 
MOREOVER, THE FOREGOING SHALL SURVIVE ANY EXPIRATION OR TERMINATION OF ANY 
LICENSE OR AGREEMENT OR ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE REGARDING THE U.S. GOVERNMENT AND DOD AGENCIES: This material is 
provided with "RESTRICTED RIGHTS" and/or "LIMITED RIGHTS" as applicable to 
computer software and technical data, respectively. Use, duplication, 
distribution or disclosure by the U.S. Government and/or DOD agencies is 
subject to the full extent of restrictions in all applicable regulations, 
including those found at FAR52.227 and DFARS252.227 et seq. and any successor 
regulations thereof. Use of this material by the U.S. Government and/or DOD 
agencies is acknowledgment of the proprietary rights of any copyright holders 
and contributors, including those of Advanced Micro Devices, Inc., as well as 
the provisions of FAR52.227-14 through 23 regarding privately developed and/or 
commercial computer software.

This license forms the entire agreement regarding the subject matter hereof and 
supersedes all proposals and prior discussions and writings between the parties 
with respect thereto. This license does not affect any ownership, rights, title,
or interest in, or relating to, this material. No terms of this license can be 
modified or waived, and no breach of this license can be excused, unless done 
so in a writing signed by all affected parties. Each term of this license is 
separately enforceable. If any term of this license is determined to be or 
becomes unenforceable or illegal, such term shall be reformed to the minimum 
extent necessary in order for this license to remain in effect in accordance 
with its terms as modified by such reformation. This license shall be governed 
by and construed in accordance with the laws of the State of Texas without 
regard to rules on conflicts of law of any state or jurisdiction or the United 
Nations Convention on the International Sale of Goods. All disputes arising out 
of this license shall be subject to the jurisdiction of the federal and state 
courts in Austin, Texas, and all defenses are hereby waived concerning personal 
jurisdiction and venue of these courts.

============================================================ */

/*
 * For a description of the algorithm and the terms used, please see the
 * documentation for this sample.
 */

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable 
// #pragma OPENCL EXTENSION cl_amd_printf : enable

#define RADIX 8
#define RADICES (1 << RADIX) // Number of elemets handled by each work-item

/**
 * @brief   Calculates block-histogram bin whose bin size is 256
 * @param   unsortedData    array of unsorted elements
 * @param   buckets         histogram buckets    
 * @param   shiftCount      shift count
 * @param   sharedArray     shared array for thread-histogram bins
  */
__kernel
void histogram(__global const uint* unsortedData,
               __global uint* buckets,
               uint shiftCount,
               __local ushort* sharedArray)
{
    size_t localId = get_local_id(0);
    size_t globalId = get_global_id(0);
    size_t groupId = get_group_id(0);
    size_t groupSize = get_local_size(0);

    /* Initialize shared array to zero */
    for(int i = 0; i < RADICES; ++i)
        sharedArray[localId * RADICES + i] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    
    /* Calculate thread-histograms */
    for(int i = 0; i < RADICES; ++i)
    {
        uint value = unsortedData[globalId * RADICES + i];
        value = (value >> shiftCount) & 0xFFU;
        sharedArray[localId * RADICES + value]++;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    /* Copy calculated histogram bin to global memory */
    for(int i = 0; i < RADICES; ++i)
    {
        uint bucketPos = groupId * RADICES * groupSize + localId * RADICES + i;
        buckets[bucketPos] = sharedArray[localId * RADICES + i];
    }
}

/**
 * @brief   Permutes the element to appropriate places based on
 *          prescaned buckets values
 * @param   unsortedData        array of unsorted elments
 * @param   unsortedValues      array of unsorted values
 * @param   scanedBuckets       prescaned buckets for permuations
 * @param   shiftCount          shift count
 * @param   sharedBuckets       shared array for scaned buckets
 * @param   sortedData          array for sorted elements
 * @param   sortedValues        array for the values to be sorted
 */
__kernel
void permute(__global const uint* unsortedData,
	     __global const uint* unsortedValues,
             __global const uint* scanedBuckets,
             uint shiftCount,
             __local ushort* sharedBuckets,
             __global uint* sortedData,
	     __global uint* sortedValues)
{

    size_t groupId = get_group_id(0);
    size_t localId = get_local_id(0);
    size_t globalId = get_global_id(0);
    size_t groupSize = get_local_size(0);
    
    
    /* Copy prescaned thread histograms to corresponding thread shared block */
    for(int i = 0; i < RADICES; ++i)
    {
        uint bucketPos = groupId * RADICES * groupSize + localId * RADICES + i;
        sharedBuckets[localId * RADICES + i] = scanedBuckets[bucketPos];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    /* Premute elements to appropriate location */
    for(int i = 0; i < RADICES; ++i)
    {
        uint value = unsortedData[globalId * RADICES + i];
        value = (value >> shiftCount) & 0xFFU;
        uint index = sharedBuckets[localId * RADICES + value];
        sortedData[index] = unsortedData[globalId * RADICES + i];

	// permute the values as well
	sortedValues[index] = unsortedValues[globalId * RADICES + i];

        sharedBuckets[localId * RADICES + value] = index + 1;
	barrier(CLK_LOCAL_MEM_FENCE);

    }
}
