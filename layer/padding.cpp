//
// Created by lifan on 2021/5/4.
//

//所有int8，fp16数据计算，待之后有机会添加

#include "padding.h"

namespace tinynn
{

Padding::Padding()
{
    one_blob_only = true;
    support_inplace = false;
}

int Padding::load_param(const ParamDict &pd)
{
    top = pd.get(0, 0);
    bottom = pd.get(1, 0);
    left = pd.get(2, 0);
    right = pd.get(3, 0);
    type = pd.get(4, 0);
    value = pd.get(5, 0.f);
    per_channel_pad_data_size = pd.get(6, 0);
    front = pd.get(7, 0);
    behind = pd.get(8, 0);

    if (top == -233 && bottom == -233 && left == -233 && right == -233)
    {
        one_blob_only = false;
    }
    if (top == -234 && bottom == -234 && left == -234 && right == -234)
    {
        one_blob_only = false;
    }

    return 0;
}

int Padding::load_model(const ModelBin &mb)
{
    if (per_channel_pad_data_size)
    {
        per_channel_pad_data = mb.load(per_channel_pad_data_size, 1);
    }

    return 0;
}
//对于原有元素的拷贝为什么要分成两种情况？
template<typename T>
static void copy_make_border_image(const Mat& src, Mat& dst, int top, int left, int type, T v)
{
    int w = dst.width;
    int h = dst.height;

    const T* ptr = src;
    T* outptr = dst;

    if (type == 0)
    {
        int y = 0;
        //fill top
        for (; y < top; y++)
        {
            int x = 0;
            for (; x < w; x++)
            {
                outptr[x] = v;
            }
            outptr += w;
        }
        //fill center
        for (; y < top + src.height; y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = v;
            }

            if (src.width < 12)
            {
                for (; x < left + src.width; x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, src.width * sizeof(T));
                x += src.width;
            }

            for (; x < w; x++)
            {
                outptr[x] = v;
            }

            outptr += w;
            ptr += src.width;
        }
        //fill bottom
        for (; y < h; y++)
        {
            int x = 0;
            for (; x < w; x++)
            {
                outptr[x] = v;
            }

            outptr += w;
        }
    }

    if (type == 1)
    {
        int y = 0;
        //fill top
        for (; y < top; y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = ptr[0];
            }

            if (src.width < 12)
            {
                for (; x < left + src.width; x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, src.width * sizeof(T));
                x += src.width;
            }

            for (; x < w; x++)
            {
                outptr[x] = ptr[src.width - 1];
            }
            outptr += w;
        }

        //fill center
        for (; y < top + src.height; y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = ptr[0];
            }

            if (src.width < 12)
            {
                for (; x < left + src.width; x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, src.width * sizeof(T));
                x += src.width;
            }
            for (; x < w; x++)
            {
                outptr[x] = ptr[src.width - 1];
            }
            outptr += w;
            ptr += src.width;
        }
        //fill bottom
        ptr -= src.width;
        for (; y < h; y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = ptr[0];
            }
            if (src.width < 12)
            {
                for (; x < left + src.width; x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, src.width * sizeof(T));
                x += src.width;
            }
            for (; x < w; x++)
            {
                outptr[x] = ptr[src.width - 1];
            }

            outptr += w;
        }
    }

    if (type == 2)
    {
        int y = 0;

        ptr += top * src.width;
        //fill top
        for (; y < top; y++)
        {
            int x = 0;

            for (; x < left; x++)
            {
                outptr[x] = ptr[left - x];
            }

            if (src.width < 12)
            {
                for (; x < left + src.width; x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, src.width * sizeof(T));
                x += src.width;
            }
            for (; x < w; x++)
            {
                outptr[x] = ptr[src.width - (x - left -src.width) - 2];
            }
            outptr += w;
            ptr -= src.width;
        }

        //fill center
        for (; y < top + src.height; y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = ptr[left - x];
            }
            if (src.width < 12)
            {
                for (; x < left + src.width; x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, src.width * sizeof(T));
                x += src.width;
            }
            for (; x < w; x++)
            {
                outptr[x] = ptr[src.width - (x - left - src.width) - 2];
            }
            outptr += w;
            ptr += src.width;
        }

        //fill bottom
        ptr -= 2 * src.width;
        for (; y < h; y++)
        {
            int x = 0;

            for (; x < left; x++)
            {
                outptr[x] = ptr[left - x];
            }
            if (src.width < 12)
            {
                for(; x < left + src.width; x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, src.width * sizeof(T));
                x += src.width;
            }
            for (; x < w; x++)
            {
                outptr[x] = ptr[src.width - (x - left - src.width) - 2];
            }
            outptr += w;
            ptr -= src.width;
        }
    }
}

int Padding::forward(const Mat &bottom_blob, Mat &top_blob, const Option &opt) const
{
    if (top == 0 && bottom == 0 && left == 0 && right == 0 && front == 0 && behind == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.width;
    int h = bottom_blob.height;
    int channels = bottom_blob.channel;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    int outw = left + w + right;

    if (dims == 1)
    {
        top_blob.create(outw, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elemsize == 1)
        {
            //TODO
//            copy_make_border_image<signed char>(bottom_blob, top_blob, 0, left, type, static_cast<signed char>(value));
        }
        if (elemsize == 2)
        {
            //TODO
//            copy_make_border_image<unsigned short>(bottom_blob, top_blob, 0, left, type, opt.use_fp16_storage ? float32_to_float16(value) : float32_to_bfloat16(value));

        }
        if (elemsize == 4)
            copy_make_border_image<float>(bottom_blob, top_blob, 0, left, type, value);

        return 0;
    }

    int outh = top + h + bottom;

    if (dims == 2)
    {
        top_blob.create(outw, outh, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elemsize == 1)
        {
            //TODO
//            copy_make_border_image<signed char>(bottom_blob, top_blob, top, left, type, static_cast<signed char>(value));
        }
        if (elemsize == 2)
        {
            //TODO
//            copy_make_border_image<unsigned short>(bottom_blob, top_blob, top, left, type, opt.use_fp16_storage ? float32_to_float16(value): float32_to_bfloat16(value));
        }
        if (elemsize == 4)
            copy_make_border_image<float>(bottom_blob, top_blob, top, left, type, value);

        return 0;
    }

    int outc = front + channels + behind;

    if (dims == 3)
    {
        top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < outc; ++q)
        {
            Mat borderm = top_blob.refer_channel(q);

            float pad_value = per_channel_pad_data_size ? per_channel_pad_data[q] : value;

            if ((q < front || q > front + channels) && type == 0)
            {
                if (elemsize == 1)
                {
                    //TODO
//                    borderm.fill(static_cast<signed char>(pad_value));
                }
                if (elemsize == 2)
                {
                    //TODO
//                    borderm.fill(opt.use_fp16_storage ? float32_to_float16(pad_value): float32_to_bfloat16(pad_value));
                }
                if (elemsize == 4)
                {
                    borderm.fill(pad_value);
                }
            }
            else
            {
                int q_ = q - front;

                if (type == 1)
                {
                    q_ = q_ <= 0 ? 0 : q_;
                    q_ = q_ >= channels - 1 ? channels - 1 : q_;
                }
                if (type == 2)
                {
                    q_ = abs(q_);
                    q_ = (channels - 1) - abs(q_ - (channels - 1));
                }
                const Mat m = bottom_blob.refer_channel(q_);

                if (elemsize == 1)
                {
                    //TODO
//                    copy_make_border_image<signed char>(m, borderm, top, left, type, static_cast<signed char>(pad_value));
                }
                if (elemsize == 2)
                {
                    //TODO
//                    copy_make_border_image<unsigned short>(m, borderm, top, left, type, opt.use_fp16_storage ? float32_to_float16(pad_value) : float32_to_bfloat16(pad_value));
                }
                if (elemsize == 4)
                {
                    copy_make_border_image<float>(m, borderm, top, left, type, pad_value);
                }
            }
        }
        return 0;
    }

    return 0;
}

int Padding::forward(const std::vector<Mat> &bottom_blobs, std::vector<Mat> &top_blobs, const Option &opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];

    Mat& top_blob = top_blobs[0];

    int _top;
    int _bottom;
    int _left;
    int _right;
    int _front;
    int _behind;
    {
        const int* param_data = reference_blob;
        _top = param_data[0];
        _bottom = param_data[1];
        _left = param_data[2];
        _right = param_data[3];
        _front = param_data[4];
        _behind = param_data[5];
    }
    if (_top == 0 && _bottom == 0 && _left == 0 && _right == 0 && _front == 0 && _behind == 0)
    {
        top_blob = bottom_blob;
    }
    int w = bottom_blob.width;
    int h = bottom_blob.height;
    int channels = bottom_blob.channel;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    int outw = _left + w + _right;

    if (dims == 1)
    {
        top_blob.create(outw, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elemsize == 1)
        {
            //TODO
//            copy_make_border_image<signed char>(bottom_blob, top_blob, 0, _left, type, static_cast<signed char>(value));
        }
        if (elemsize == 2)
        {
            //TODO
//            copy_make_border_image<unsigned short>(bottom_blob, top_blob, 0, _left, type, opt.use_fp16_storage ? float32_to_float16(value) : float32_to_bfloat16(value));
        }
        if (elemsize == 4)
            copy_make_border_image<float>(bottom_blob, top_blob, 0, _left, type, value);

        return 0;
    }

    int outh = _top + h + _bottom;

    if (dims == 2)
    {
        top_blob.create(outw, outh, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elemsize == 1)
        {
            //TODO
//            copy_make_border_image<signed char>(bottom_blob, top_blob, _top, _left, type, static_cast<signed char>(value));
        }
        if (elemsize == 2)
        {
            //TODO
//            copy_make_border_image<unsigned short>(bottom_blob, top_blob, _top, _left, type, opt.use_fp16_storage ? float32_to_float16(value): float32_to_bfloat16(value));
        }
        if (elemsize == 4)
            copy_make_border_image<float>(bottom_blob, top_blob, _top, _left, type, value);

        return 0;
    }

    int outc = _front + channels + _behind;

    if (dims == 3)
    {
        top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < outc; ++q)
        {
            Mat borderm = top_blob.refer_channel(q);

            float pad_value = per_channel_pad_data_size ? per_channel_pad_data[q] : value;

            if ((q < _front || q > _front + channels) && type == 0)
            {
                if (elemsize == 1)
                {
                    //TODO
//                    borderm.fill(static_cast<signed char>(pad_value));
                }
                if (elemsize == 2)
                {
                    //TODO
//                    borderm.fill(opt.use_fp16_storage ? float32_to_float16(pad_value): float32_to_bfloat16(pad_value));
                }
                if (elemsize == 4)
                {
                    borderm.fill(pad_value);
                }
            }
            else
            {
                int q_ = q - _front;

                if (type == 1)
                {
                    q_ = q_ <= 0 ? 0 : q_;
                    q_ = q_ >= channels - 1 ? channels - 1 : q_;
                }
                if (type == 2)
                {
                    q_ = abs(q_);
                    q_ = (channels - 1) - abs(q_ - (channels - 1));
                }
                const Mat m = bottom_blob.refer_channel(q_);

                if (elemsize == 1)
                {
                    //TODO
//                    copy_make_border_image<signed char>(m, borderm, _top, _left, type, static_cast<signed char>(pad_value));
                }
                if (elemsize == 2)
                {
                    //TODO
//                    copy_make_border_image<unsigned short>(m, borderm, _top, _left, type, opt.use_fp16_storage ? float32_to_float16(pad_value) : float32_to_bfloat16(pad_value));
                }
                if (elemsize == 4)
                {
                    copy_make_border_image<float>(m, borderm, _top, _left, type, pad_value);
                }
            }
        }
        return 0;
    }

    return 0;
}


}