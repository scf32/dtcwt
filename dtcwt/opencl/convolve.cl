// Required defines:

// width of filter *MUST BE ODD*
#ifndef FILTER_WIDTH
#   error "Filter width must be defined"
#endif
#if (FILTER_WIDTH & 0x1) != 1
#   error "Filter width must be odd"
#endif

// work group size as would be returned by get_local_size(0)
#ifndef LOCAL_SIZE_0
#   error "Local size 0 must be defined"
#endif
// work group size as would be returned by
// get_local_size(1) * ... * get_local_size(get_work_dim()-1)
#ifndef LOCAL_SIZE_REST
#   error "Local size 1...rest must be defined"
#endif

// datatype of input
#ifndef INPUT_TYPE
#   error "Input data type must be defined"
#endif

// Derived values
#define FILTER_HALF_WIDTH ((FILTER_WIDTH-1)>>1)
#define LOCAL_CACHE_WIDTH ((2*FILTER_HALF_WIDTH)+LOCAL_SIZE_0)

// Return a linear offset to the specified element
int index(int4 coord, int4 strides) {
    // unfortunately dot() is only defined for floating point types
    int4 prod = coord * strides;
    return prod.x + prod.y + prod.z + prod.w;
}

// magic function to reflect the sampling co-ordinate about the
// *outer edges* of pixel co-ordinates x_min, x_max. The output will
// always be in the range (x_min, x_max].
int4 reflect(int4 x, int4 x_min, int4 x_max)
{
    int4 rng = x_max - x_min;
    int4 rng_by_2 = 2 * rng;
    int4 mod = (x - x_min) % rng_by_2;
    int4 normed_mod = select(mod, mod + rng_by_2, mod < 0);
    return select(normed_mod, rng_by_2 - normed_mod - (int4)(1,1,1,1), normed_mod >= rng) + x_min;
}

// Convolve along first axis. To avoid this swizzle input_{...} appropriately.
// Strides, offsets, skips and shapes are measured in units of INPUT_TYPE.
//
// The shape of the region of input pixels to process is specified by
// output_shape. *THIS IS NOT INCLUDING SKIP*. Processing a region of (N,M) in
// shape with a skip of (2,3) will result in (2N,3M) pixels of input and output
// being touched.
//
// Processing will start at offset input_offset from zero.
// Neighbouring pixels are assumed to occur input_skip pixels along each
// dimension (this would usually be 1) with each dimension requiring
// input_strides to advance. Note that, for offsets which are multiples of
// input_skip, setting input_offset = input_offset / input_skip and
// input_strides = input_strides / input_skip and then input_skip = 1 will have
// the same effect.
//
// IMPORTANT: Setting input_offset, output_offset or output_shape such that
// pixels in an invalid region are accessed is undefined and not checked for!
__kernel void convolve(
    __constant float* filter_kernel, int n_convolutions, int4 pixels_to_write,
    __global INPUT_TYPE* input, int input_start,
    int4 input_offset, int4 input_shape, int4 input_skip, int4 input_strides,
    __global INPUT_TYPE* output, int output_start,
    int4 output_offset, int4 output_shape, int4 output_skip, int4 output_strides)
{
    input += input_start;
    output += output_start;

    // Create an appropriately sized region of local memory which can hold the
    // input plus some apron.
    __local INPUT_TYPE input_cache[LOCAL_CACHE_WIDTH*LOCAL_SIZE_REST];

    // Compute upper-left corner of this work group in input and output
    int4 group_coord = (int4)(
        get_group_id(0) * get_local_size(0), get_group_id(1) * get_local_size(1),
        0, 0
    );
    int4 input_origin = input_offset + input_skip * group_coord;
    int4 output_origin = output_offset + output_skip * group_coord;
    int4 local_coord = (int4)(get_local_id(0), get_local_id(1), 0, 0);

    // This is the output pixel this work item should write to
    int4 output_coord = output_origin + output_skip*local_coord;

    // This is the corresponding input pixel to read from
    int4 input_coord = input_origin + input_skip*local_coord;

    for(int w=0; w<pixels_to_write.w; ++w, ++output_coord.w, ++input_coord.w)
    {
        input_coord.z = input_origin.z;
        output_coord.z = output_origin.z;
        for(int z=0; z<pixels_to_write.z; ++z, ++output_coord.z, ++input_coord.z)
        {
            // Copy input into cache
            input_cache[get_local_id(0) + FILTER_HALF_WIDTH +
                LOCAL_CACHE_WIDTH * get_local_id(1)] = input[
                    index(reflect(input_coord, 0, input_shape), input_strides)];
            if(get_local_id(0) < FILTER_HALF_WIDTH) {
                input_cache[get_local_id(0) +
                    LOCAL_CACHE_WIDTH * get_local_id(1)] = input[index(
                        reflect(input_coord - input_skip*(int4)(FILTER_HALF_WIDTH,0,0,0),
                            0, input_shape),
                        input_strides)];
            }
            if(get_local_id(0) >= get_local_size(0) - FILTER_HALF_WIDTH) {
                input_cache[get_local_id(0) + 2*(FILTER_HALF_WIDTH) +
                    LOCAL_CACHE_WIDTH * get_local_id(1)] = input[index(
                        reflect(input_coord + input_skip*(int4)(FILTER_HALF_WIDTH,0,0,0),
                            0, input_shape),
                        input_strides)];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // abort if we're writing outside the valid region. Do so now
            // because we may still have read something important into the
            // input cache.
            if(any(output_coord < 0) || any(output_coord >= output_shape)) {
                continue;
            }

            for(int conv_idx=0; conv_idx < n_convolutions; ++conv_idx) {
                // generate output pixel value
                float filter_tap;
                INPUT_TYPE output_value = 0.f, input_value;
                for(int f_idx=0; f_idx<FILTER_WIDTH; ++f_idx) {
                    input_value = input_cache[
                        get_local_id(0) + f_idx +
                        get_local_id(1) * LOCAL_CACHE_WIDTH
                    ];
                    //input_value = 1.f;
                    filter_tap = filter_kernel[f_idx + conv_idx*FILTER_WIDTH];
                    output_value += input_value * filter_tap;
                }

                // write output pixel value
                output[n_convolutions * index(output_coord, output_strides) + conv_idx] =
                    output_value;
            }
        }
    }
}

__kernel void copy_with_sampling(
    int4 pixels_to_write,
    __global INPUT_TYPE* input, int input_start,
    int4 input_offset, int4 input_shape, int4 input_skip, int4 input_strides,
    __global INPUT_TYPE* output, int output_start,
    int4 output_offset, int4 output_shape, int4 output_skip, int4 output_strides)
{
    input += input_start;
    output += output_start;

    // Compute upper-left corner of this work group in input and output
    int4 group_coord = (int4)(
        get_group_id(0) * get_local_size(0), get_group_id(1) * get_local_size(1),
        0, 0
    );
    int4 local_coord = (int4)(get_local_id(0), get_local_id(1), 0, 0);
    int4 input_origin = input_offset + input_skip * group_coord;
    int4 output_origin = output_offset + output_skip * group_coord;

    // This is the output pixel this work item should write to
    int4 output_coord = output_origin + output_skip*local_coord;

    // Abort on invalid output coord
    if(any(output_coord < 0) || any(output_coord >= output_shape)) { return; }

    // This is the corresponding input pixel to read from
    int4 input_coord = input_origin + input_skip*local_coord;

    for(int w=0; w<pixels_to_write.w; ++w, ++output_coord.w, ++input_coord.w)
    {
        input_coord.z = input_origin.z;
        output_coord.z = output_origin.z;
        for(int z=0; z<pixels_to_write.z; ++z, ++output_coord.z, ++input_coord.z)
        {
            // Reflect input coord
            int4 sample_coord = reflect(input_coord, 0, input_shape);

            // Copy input to output
            output[index(output_coord, output_strides)] =
                input[index(sample_coord, input_strides)];
        }
    }
}

void q2c(INPUT_TYPE a, INPUT_TYPE b, INPUT_TYPE c, INPUT_TYPE d,
         INPUT_TYPE *z1real, INPUT_TYPE *z1imag,
         INPUT_TYPE *z2real, INPUT_TYPE *z2imag)
{
    const float sqrt_half = sqrt(0.5);
    INPUT_TYPE preal = a * sqrt_half, pimag = b * sqrt_half;
    INPUT_TYPE qreal = d * sqrt_half, qimag = -c * sqrt_half;

    *z1real = preal - qreal;
    *z1imag = pimag - qimag;
    *z2real = preal + qreal;
    *z2imag = pimag + qimag;
}

// Input is tightly packed C-style array of 4 INPUT_TYPE elements per pixel.
// Lo output is tightly packed C-style array of INPUT_TYPE elements per pixel.
// Hi output is tightly packed C-style half-size array of 6 pairs of INPUT_TYPE elements per pixel.
// Each work item works on one element of the highpass output => 2x2 block of lowpass output and input.
//
// Shape is (height, width). Co-ordinates are row-column.
__kernel void copy_level1_output(int4 hi_shape,
    __global INPUT_TYPE* input, int input_start,
    __global INPUT_TYPE* lo_output, int lo_output_start,
    __global INPUT_TYPE* hi_output, int hi_output_start)
{
    input += input_start;
    lo_output += lo_output_start;
    hi_output += hi_output_start;

    // Compute upper-left corner of this work group in input and output
    int4 group_coord = (int4)(
        get_group_id(0) * get_local_size(0), get_group_id(1) * get_local_size(1),
        0, 0
    );
    int4 local_coord = (int4)(get_local_id(0), get_local_id(1), 0, 0);

    // Where are we in highpass output?
    int4 hi_coord = group_coord + local_coord;

    // Abort if in invalid area.
    if(any(hi_coord < 0) || any(hi_coord >= hi_shape)) { return; }

    // Advance input pointer to start of input block
    int input_row_stride = 4*2*hi_shape.s1;
    input += input_row_stride*2*hi_coord.s0;    // row
    input += 4*2*hi_coord.s1;                   // column

    // Advance output pointers to start of output.
    int lo_row_stride = 2*hi_shape.s1;
    lo_output += lo_row_stride*2*hi_coord.s0;   // row
    lo_output += 2*hi_coord.s1;                 // col
    int hi_row_stride = 12*hi_shape.s1;
    hi_output += hi_row_stride*hi_coord.s0;     // row
    hi_output += 12*hi_coord.s1;                // col

    // Copy 4 lowpass pixels
    lo_output[0] = input[0];
    lo_output[1] = input[4];
    lo_output[0+lo_row_stride] = input[0+input_row_stride];
    lo_output[1+lo_row_stride] = input[4+input_row_stride];

    INPUT_TYPE a, b, c, d, z1r, z1i, z2r, z2i;

    // lohi
    a = input[1]; b = input[5];
    c = input[1+input_row_stride]; d = input[5+input_row_stride];
    q2c(a,b,c,d,&z1r,&z1i,&z2r,&z2i);
    hi_output[0]=z1r; hi_output[1]=z1i; hi_output[10]=z2r; hi_output[11]=z2i;

    // hilo
    a = input[2]; b = input[6];
    c = input[2+input_row_stride]; d = input[6+input_row_stride];
    q2c(a,b,c,d,&z1r,&z1i,&z2r,&z2i);
    hi_output[4]=z1r; hi_output[5]=z1i; hi_output[6]=z2r; hi_output[7]=z2i;

    // hihi
    a = input[3]; b = input[7];
    c = input[3+input_row_stride]; d = input[7+input_row_stride];
    q2c(a,b,c,d,&z1r,&z1i,&z2r,&z2i);
    hi_output[2]=z1r; hi_output[3]=z1i; hi_output[8]=z2r; hi_output[9]=z2i;
}
