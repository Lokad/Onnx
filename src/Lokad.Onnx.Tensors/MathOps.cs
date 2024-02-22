using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using static Lokad.Onnx.MathOps;

namespace Lokad.Onnx
{
    internal class cs_BLAS
    {
        public static void XERBLA(string srname, int info)
        {
            // This is a special version of XERBLA to be used only as part of the test program for testing error exits from the Level 2 BLAS routines.
            // XERBLA is an error handler for the Level 2 BLAS routines. It is called by the Level 2 BLAS routines if an input parameter is invalid.
            /* .. Scalars in Common ..
            INTEGER INFOT, NOUT
            LOGICAL LERR, OK
            CHARACTER*6 SRNAMT*/
            // Executable Statements
            //lerr = true;
            //nout.WriteLine("in:srname={0} info={1} :infot={2} srnamt={3}", srname, info, infot, srnamt);
            throw new Exception(string.Format("in:srname={0} info={1}", srname, info));

        }
        public static void DGEMM(string transa, string transb, int m, int n, int k, double alpha, double[,] a, int lda, double[,] b, int ldb, double beta,
ref double[,] c, int ldc)
        {
            //DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
            //DOUBLE PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
            // DGEMM performs one of the matrix-matrix operations
            //
            // C := alpha*op( A )*op( B ) + beta*C,
            //
            // where op( X ) is one of
            //
            // op( X ) = X or op( X ) = X**T,
            //
            // alpha and beta are scalars, and A, B and C are matrices, with op( A ) an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
            // On entry, TRANSA specifies the form of op( A ) to be used in the matrix multiplication as follows:
            // TRANSA = 'N' or 'n', op( A ) = A.
            // TRANSA = 'T' or 't', op( A ) = A**T.
            // TRANSA = 'C' or 'c', op( A ) = A**T.
            // On entry, TRANSB specifies the form of op( B ) to be used in the matrix multiplication as follows:
            // TRANSB = 'N' or 'n', op( B ) = B.
            // TRANSB = 'T' or 't', op( B ) = B**T.
            // TRANSB = 'C' or 'c', op( B ) = B**T.
            // On entry, M specifies the number of rows of the matrix op( A ) and of the matrix C. M must be at least zero.
            // On entry, N specifies the number of columns of the matrix op( B ) and the number of columns of the matrix C. N must be at least zero.
            // On entry, K specifies the number of columns of the matrix op( A ) and the number of rows of the matrix op( B ). K must be at least zero.
            // On entry, ALPHA specifies the scalar alpha.
            // A is DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is k when TRANSA = 'N' or 'n', and is m otherwise. Before entry with
            // TRANSA = 'N' or 'n', the leading m by k
            // part of the array A must contain the matrix A, otherwise the leading k by m part of the array A must contain the matrix A.
            // On entry, LDA specifies the first dimension of A as declared in the calling (sub) program. When TRANSA = 'N' or 'n' then LDA must be at
            // least max(1, m), otherwise LDA must be at
            // least max( 1, k ).
            // B is DOUBLE PRECISION array of DIMENSION ( LDB, kb ), where kb is n when TRANSB = 'N' or 'n', and is k otherwise. Before entry with
            // TRANSB = 'N' or 'n', the leading k by n
            // part of the array B must contain the matrix B, otherwise the leading n by k part of the array B must contain the matrix B.
            // On entry, LDB specifies the first dimension of B as declared in the calling (sub) program. When TRANSB = 'N' or 'n' then LDB must be at
            // least max(1, k), otherwise LDB must be at
            // least max( 1, n ).
            // On entry, BETA specifies the scalar beta. When BETA is supplied as zero then C need not be set on input.
            // C is DOUBLE PRECISION array of DIMENSION ( LDC, n ). Before entry, the leading m by n part of the array C must contain the matrix C,
            // except when beta is zero, in which
            // case C need not be set on entry. On exit, the array C is overwritten by the m by n matrix ( alpha*op( A )*op( B ) + beta*C ).
            // On entry, LDC specifies the first dimension of C as declared in the calling (sub) program. LDC must be at least max( 1, m ).
            // Level 3 Blas routine.
            // DOUBLE PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
            // Local Scalars
            double temp;
            int info, ncola, nrowa, nrowb;
            bool nota, notb;
            // Parameters
            double one = 1.0;
            double zero = 0.0;
            // Set NOTA and NOTB as true if A and B respectively are not transposed and set NROWA, NCOLA and NROWB as the number of rows and
            //columns of A and the number of rows of B respectively.
            nota = (transa.Substring(0, 1).ToUpper() == "N");
            notb = (transb.Substring(0, 1).ToUpper() == "N");
            if (nota)
            {
                nrowa = m;
                ncola = k;
            }
            else
            {
                nrowa = k;
                ncola = m;
            }
            if (notb)
            {
                nrowb = k;
            }
            else
            {
                nrowb = n;
            }
            // Test the input parameters
            info = 0;
            if (!nota && !(transa.Substring(0, 1).ToUpper() == "C") && !(transa.Substring(0, 1).ToUpper() == "T"))
            {
                info = 1;
            }
            else if (!notb && !(transb.Substring(0, 1).ToUpper() == "C") && !(transb.Substring(0, 1).ToUpper() == "T"))
            {
                info = 2;
            }
            else if (m < 0)
            {
                info = 3;
            }
            else if (n < 0)
            {
                info = 4;
            }
            else if (k < 0)
            {
                info = 5;
            }
            else if (lda < System.Math.Max(1, nrowa))
            {
                info = 8;
            }
            else if (ldb < System.Math.Max(1, nrowb))
            {
                info = 10;
            }
            else if (ldc < System.Math.Max(1, m))
            {
                info = 13;
            }
            if (info != 0)
            {
                XERBLA("DGEMM ", info);
                return;
            }
            // Quick return if possible
            if ((m == 0) || (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one)))
            {
                return;
            }
            // And if alpha.eq.zero.
            if (alpha == zero)
            {
                if (beta == zero)
                {
                    for (int j = 1; j <= n; j = j + 1)
                    {
                        for (int i = 1; i <= m; i = i + 1)
                        {
                            c[i - 1, j - 1] = zero;
                        }
                    }
                }
                else
                {
                    for (int j = 1; j <= n; j = j + 1)
                    {
                        for (int i = 1; i <= m; i = i + 1)
                        {
                            c[i - 1, j - 1] = beta * c[i - 1, j - 1];
                        }
                    }
                }
                return;
            }
            // Start the operations.
            if (notb)
            {
                if (nota)
                {
                    // Form C := alpha*A*B + beta*C
                    for (int j = 1; j <= n; j = j + 1)
                    {
                        if (beta == zero)
                        {
                            for (int i = 1; i <= m; i = i + 1)
                            {
                                c[i - 1, j - 1] = zero;
                            }
                        }
                        else if (beta != one)
                        {
                            for (int i = 1; i <= m; i = i + 1)
                            {
                                c[i - 1, j - 1] = beta * c[i - 1, j - 1];
                            }
                        }
                        for (int l = 1; l <= k; l = l + 1)
                        {
                            if (b[l - 1, j - 1] != zero)
                            {
                                temp = alpha * b[l - 1, j - 1];
                                for (int i = 1; i <= m; i = i + 1)
                                {
                                    c[i - 1, j - 1] = c[i - 1, j - 1] + temp * a[i - 1, l - 1];
                                }
                            }
                        }
                    }
                }
                else
                {
                    // Form C := alpha*A**T*B + beta*C
                    for (int j = 1; j <= n; j = j + 1)
                    {
                        for (int i = 1; i <= m; i = i + 1)
                        {
                            temp = zero;
                            for (int l = 1; l <= k; l = l + 1)
                            {
                                temp = temp + a[l - 1, i - 1] * b[l - 1, j - 1];
                            }
                            if (beta == zero)
                            {
                                c[i - 1, j - 1] = alpha * temp;
                            }
                            else
                            {
                                c[i - 1, j - 1] = alpha * temp + beta * c[i - 1, j - 1];
                            }
                        }
                    }
                }
            }
            else
            {
                if (nota)
                {
                    // Form C := alpha*A*B**T + beta*C
                    //170
                    for (int j = 1; j <= n; j = j + 1)
                    {
                        if (beta == zero)
                        {
                            for (int i = 1; i <= m; i = i + 1)
                            {
                                c[i - 1, j - 1] = zero;
                            }
                        }
                        else if (beta != one)
                        {
                            for (int i = 1; i <= m; i = i + 1)
                            {
                                c[i - 1, j - 1] = beta * c[i - 1, j - 1];
                            }
                        }
                        for (int l = 1; l <= k; l = l + 1)
                        {
                            if (b[j - 1, l - 1] != zero)
                            {
                                temp = alpha * b[j - 1, l - 1];
                                for (int i = 1; i <= m; i = i + 1)
                                {
                                    c[i - 1, j - 1] = c[i - 1, j - 1] + temp * a[i - 1, l - 1];
                                }
                            }
                        }
                    } //170
                }
                else
                {
                    // Form C := alpha*A**T*B**T + beta*C
                    for (int j = 1; j <= n; j = j + 1)
                    {
                        for (int i = 1; i <= m; i = i + 1)
                        {
                            temp = zero;
                            for (int l = 1; l <= k; l = l + 1)
                            {
                                temp = temp + a[l - 1, i - 1] * b[j - 1, l - 1];
                            }
                            if (beta == zero)
                            {
                                c[i - 1, j - 1] = alpha * temp;
                            }
                            else
                            {
                                c[i - 1, j - 1] = alpha * temp + beta * c[i - 1, j - 1];
                            }
                        }
                    }
                }
            }
        }

        public static void DGEMM(string transa, string transb, int m, int n, int k, float alpha, float[,] a, int lda, float[,] b, int ldb, float beta,
ref float[,] c, int ldc)
        {
            //DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
            //float PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
            // DGEMM performs one of the matrix-matrix operations
            //
            // C := alpha*op( A )*op( B ) + beta*C,
            //
            // where op( X ) is one of
            //
            // op( X ) = X or op( X ) = X**T,
            //
            // alpha and beta are scalars, and A, B and C are matrices, with op( A ) an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
            // On entry, TRANSA specifies the form of op( A ) to be used in the matrix multiplication as follows:
            // TRANSA = 'N' or 'n', op( A ) = A.
            // TRANSA = 'T' or 't', op( A ) = A**T.
            // TRANSA = 'C' or 'c', op( A ) = A**T.
            // On entry, TRANSB specifies the form of op( B ) to be used in the matrix multiplication as follows:
            // TRANSB = 'N' or 'n', op( B ) = B.
            // TRANSB = 'T' or 't', op( B ) = B**T.
            // TRANSB = 'C' or 'c', op( B ) = B**T.
            // On entry, M specifies the number of rows of the matrix op( A ) and of the matrix C. M must be at least zero.
            // On entry, N specifies the number of columns of the matrix op( B ) and the number of columns of the matrix C. N must be at least zero.
            // On entry, K specifies the number of columns of the matrix op( A ) and the number of rows of the matrix op( B ). K must be at least zero.
            // On entry, ALPHA specifies the scalar alpha.
            // A is float PRECISION array of DIMENSION ( LDA, ka ), where ka is k when TRANSA = 'N' or 'n', and is m otherwise. Before entry with
            // TRANSA = 'N' or 'n', the leading m by k
            // part of the array A must contain the matrix A, otherwise the leading k by m part of the array A must contain the matrix A.
            // On entry, LDA specifies the first dimension of A as declared in the calling (sub) program. When TRANSA = 'N' or 'n' then LDA must be at
            // least max(1, m), otherwise LDA must be at
            // least max( 1, k ).
            // B is float PRECISION array of DIMENSION ( LDB, kb ), where kb is n when TRANSB = 'N' or 'n', and is k otherwise. Before entry with
            // TRANSB = 'N' or 'n', the leading k by n
            // part of the array B must contain the matrix B, otherwise the leading n by k part of the array B must contain the matrix B.
            // On entry, LDB specifies the first dimension of B as declared in the calling (sub) program. When TRANSB = 'N' or 'n' then LDB must be at
            // least max(1, k), otherwise LDB must be at
            // least max( 1, n ).
            // On entry, BETA specifies the scalar beta. When BETA is supplied as zero then C need not be set on input.
            // C is float PRECISION array of DIMENSION ( LDC, n ). Before entry, the leading m by n part of the array C must contain the matrix C,
            // except when beta is zero, in which
            // case C need not be set on entry. On exit, the array C is overwritten by the m by n matrix ( alpha*op( A )*op( B ) + beta*C ).
            // On entry, LDC specifies the first dimension of C as declared in the calling (sub) program. LDC must be at least max( 1, m ).
            // Level 3 Blas routine.
            // float PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
            // Local Scalars
            float temp;
            int info, ncola, nrowa, nrowb;
            bool nota, notb;
            // Parameters
            float one = 1.0f;
            float zero = 0.0f;
            // Set NOTA and NOTB as true if A and B respectively are not transposed and set NROWA, NCOLA and NROWB as the number of rows and
            //columns of A and the number of rows of B respectively.
            nota = (transa.Substring(0, 1).ToUpper() == "N");
            notb = (transb.Substring(0, 1).ToUpper() == "N");
            if (nota)
            {
                nrowa = m;
                ncola = k;
            }
            else
            {
                nrowa = k;
                ncola = m;
            }
            if (notb)
            {
                nrowb = k;
            }
            else
            {
                nrowb = n;
            }
            // Test the input parameters
            info = 0;
            if (!nota && !(transa.Substring(0, 1).ToUpper() == "C") && !(transa.Substring(0, 1).ToUpper() == "T"))
            {
                info = 1;
            }
            else if (!notb && !(transb.Substring(0, 1).ToUpper() == "C") && !(transb.Substring(0, 1).ToUpper() == "T"))
            {
                info = 2;
            }
            else if (m < 0)
            {
                info = 3;
            }
            else if (n < 0)
            {
                info = 4;
            }
            else if (k < 0)
            {
                info = 5;
            }
            else if (lda < System.Math.Max(1, nrowa))
            {
                info = 8;
            }
            else if (ldb < System.Math.Max(1, nrowb))
            {
                info = 10;
            }
            else if (ldc < System.Math.Max(1, m))
            {
                info = 13;
            }
            if (info != 0)
            {
                XERBLA("DGEMM ", info);
                return;
            }
            // Quick return if possible
            if ((m == 0) || (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one)))
            {
                return;
            }
            // And if alpha.eq.zero.
            if (alpha == zero)
            {
                if (beta == zero)
                {
                    for (int j = 1; j <= n; j = j + 1)
                    {
                        for (int i = 1; i <= m; i = i + 1)
                        {
                            c[i - 1, j - 1] = zero;
                        }
                    }
                }
                else
                {
                    for (int j = 1; j <= n; j = j + 1)
                    {
                        for (int i = 1; i <= m; i = i + 1)
                        {
                            c[i - 1, j - 1] = beta * c[i - 1, j - 1];
                        }
                    }
                }
                return;
            }
            // Start the operations.
            if (notb)
            {
                if (nota)
                {
                    // Form C := alpha*A*B + beta*C
                    for (int j = 1; j <= n; j = j + 1)
                    {
                        if (beta == zero)
                        {
                            for (int i = 1; i <= m; i = i + 1)
                            {
                                c[i - 1, j - 1] = zero;
                            }
                        }
                        else if (beta != one)
                        {
                            for (int i = 1; i <= m; i = i + 1)
                            {
                                c[i - 1, j - 1] = beta * c[i - 1, j - 1];
                            }
                        }
                        for (int l = 1; l <= k; l = l + 1)
                        {
                            if (b[l - 1, j - 1] != zero)
                            {
                                temp = alpha * b[l - 1, j - 1];
                                for (int i = 1; i <= m; i = i + 1)
                                {
                                    c[i - 1, j - 1] = c[i - 1, j - 1] + temp * a[i - 1, l - 1];
                                }
                            }
                        }
                    }
                }
                else
                {
                    // Form C := alpha*A**T*B + beta*C
                    for (int j = 1; j <= n; j = j + 1)
                    {
                        for (int i = 1; i <= m; i = i + 1)
                        {
                            temp = zero;
                            for (int l = 1; l <= k; l = l + 1)
                            {
                                temp = temp + a[l - 1, i - 1] * b[l - 1, j - 1];
                            }
                            if (beta == zero)
                            {
                                c[i - 1, j - 1] = alpha * temp;
                            }
                            else
                            {
                                c[i - 1, j - 1] = alpha * temp + beta * c[i - 1, j - 1];
                            }
                        }
                    }
                }
            }
            else
            {
                if (nota)
                {
                    // Form C := alpha*A*B**T + beta*C
                    //170
                    for (int j = 1; j <= n; j = j + 1)
                    {
                        if (beta == zero)
                        {
                            for (int i = 1; i <= m; i = i + 1)
                            {
                                c[i - 1, j - 1] = zero;
                            }
                        }
                        else if (beta != one)
                        {
                            for (int i = 1; i <= m; i = i + 1)
                            {
                                c[i - 1, j - 1] = beta * c[i - 1, j - 1];
                            }
                        }
                        for (int l = 1; l <= k; l = l + 1)
                        {
                            if (b[j - 1, l - 1] != zero)
                            {
                                temp = alpha * b[j - 1, l - 1];
                                for (int i = 1; i <= m; i = i + 1)
                                {
                                    c[i - 1, j - 1] = c[i - 1, j - 1] + temp * a[i - 1, l - 1];
                                }
                            }
                        }
                    } //170
                }
                else
                {
                    // Form C := alpha*A**T*B**T + beta*C
                    for (int j = 1; j <= n; j = j + 1)
                    {
                        for (int i = 1; i <= m; i = i + 1)
                        {
                            temp = zero;
                            for (int l = 1; l <= k; l = l + 1)
                            {
                                temp = temp + a[l - 1, i - 1] * b[j - 1, l - 1];
                            }
                            if (beta == zero)
                            {
                                c[i - 1, j - 1] = alpha * temp;
                            }
                            else
                            {
                                c[i - 1, j - 1] = alpha * temp + beta * c[i - 1, j - 1];
                            }
                        }
                    }
                }
            }
        }
    }

    public class MathOps
    {
        public struct PadInfo
        {
            public int h;
            public int w;
            public int top;
            public int left;
            public int right;
            public int bottom;
        }

        public enum PadType
        {
            Valid,
            SameUpper,
            SameLower,
            Value
        }

        public struct Conv2DOutputInfo
        {
            public PadInfo PadInfo;
            public int[] Shape;
        }

        public static Conv2DOutputInfo GetConv2DOutputInfo(PadType pad, int inHeight, int inWidth, int strideHeight, int strideWidth, int filterHeight, int filterWidth, int? padValue = null)
        {
            var padInfo = new PadInfo();
            var outHeight = 0;
            var outWidth = 0;
            switch (pad)
            {
                case PadType.Valid:
                    padInfo.bottom = 0;
                    padInfo.left = 0;
                    padInfo.right = 0;
                    padInfo.top = 0;
                    outHeight = (int)Math.Ceiling((inHeight - filterHeight + 1d) / strideHeight);
                    outWidth = (int)Math.Ceiling((inWidth - filterWidth + 1d) / strideWidth);
                    padInfo.h = 0;
                    padInfo.w = 0;
                    break;

                case PadType.SameUpper:
                case PadType.SameLower:
                    outHeight = (int)Math.Ceiling(inHeight / (float)strideHeight);
                    outWidth = (int)Math.Ceiling(inWidth / (float)strideWidth);

                    var padAlongHeight = (outHeight - 1) * strideHeight + filterHeight - inHeight;
                    var padAlongWidth = (outWidth - 1) * strideWidth + filterWidth - inWidth;
                    var top = (int)Math.Floor(padAlongHeight / 2f);
                    var bottom = (int)padAlongHeight - top;
                    var left = (int)Math.Floor(padAlongWidth / 2f);
                    var right = (int)padAlongWidth - left;

                    padInfo.bottom = bottom;
                    padInfo.left = left;
                    padInfo.right = right;
                    padInfo.top = top;

                    padInfo.h = padAlongHeight;
                    padInfo.w = padAlongWidth;
                    break;

                case PadType.Value:
                    if (padValue == null) throw new ArgumentNullException(nameof(padValue));
                    padInfo.bottom = padValue.Value;
                    padInfo.left = padValue.Value;
                    padInfo.right = padValue.Value;
                    padInfo.top = padValue.Value;
                    padInfo.h = padInfo.top + padInfo.bottom;
                    padInfo.w = padInfo.right + padInfo.left;
                    var outShape = GetConv2DOutputShape(new int[] { inHeight, inWidth}, filterHeight, filterWidth, strideHeight, strideWidth, padInfo.h, padInfo.w);
                    outHeight = outShape[0];
                    outWidth = outShape[1];
                    break;
            }
            return new Conv2DOutputInfo { PadInfo = padInfo, Shape = new int[] { outHeight, outWidth } };
        }

        public static int[] GetConv2DOutputShape(int[] inputShape, int kernelHeight, int kernelWidth, int strideY, int strideX, int padY, int padX)
        {
            var outputHeight = (int) Math.Floor((inputShape[0] - kernelHeight + padY * 1.0f) / strideY) + 1;
            var outputWidth = (int) Math.Floor((inputShape[1] - kernelWidth + padX * 1.0f) / strideX) + 1;
            return new int[] { outputHeight, outputWidth};
        }

        public static int GetConv2DDefaultPad(int[] inputShape, int fieldSize, int stride, int dilation = 1) => 
            (int) Math.Floor(((float)inputShape[0] * (stride - 1) - stride + GetConv2DEffectiveFilterSize(fieldSize, dilation)) / 2);
        
        public static int GetConv2DEffectiveFilterSize(int filterSize, int dilation) => dilation <= 1 ? filterSize : filterSize + (filterSize - 1) * (dilation - 1);
        
        /// <summary>
        /// Image to column conversion.
        /// </summary>
        /// <param name="src">Source data.</param>
        /// <param name="srcC">Input channels.</param>
        /// <param name="srcH">Input height.</param>
        /// <param name="srcW">Input width.</param>
        /// <param name="kernelY">Kernel height.</param>
        /// <param name="kernelX">Kernel width.</param>
        /// <param name="dilationY">Dilation of the kernel by height.</param>
        /// <param name="dilationX">Dilation of the kernel by width.</param>
        /// <param name="strideY">Stride of the convolution by height.</param>
        /// <param name="strideX">Stride of the convolution by width.</param>
        /// <param name="padY">Zero padding by left side.</param>
        /// <param name="padX">Zero padding by top side.</param>
        /// <param name="padH">Zero padding by right side.</param>
        /// <param name="padW">Zero padding by bottom side.</param>
        /// <param name="buf">Buffer.</param>
        public static unsafe void Im2col(float* src,
                                  int srcC,
                                  int srcH,
                                  int srcW,
                                  int kernelY,
                                  int kernelX,
                                  int dilationY,
                                  int dilationX,
                                  int strideY,
                                  int strideX,
                                  int padY,
                                  int padX,
                                  int padH,
                                  int padW,
                                  float* buf)
        {
            int dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
            int dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
            for (int sc = 0; sc < srcC; ++sc)
            {
                var scsrcH = sc * srcH;
                for (int ky = 0; ky < kernelY; ++ky)
                {
                    int sy_ = ky * dilationY - padY;
                    for (int kx = 0; kx < kernelX; ++kx)
                    {
                        int sx_ = kx * dilationX - padX;
                        for (int dy = 0; dy < dstH; ++dy)
                        {
                            int sy = sy_ + dy * strideY;
                            if ((sy < 0) || (sy >= srcH))
                            {
                                for (int dx = 0; dx < dstW; ++dx)
                                {
                                    *buf++ = 0;
                                }
                                continue;
                            }
                            var src1 = src + (scsrcH + sy) * srcW;
                            for (int dx = 0; dx < dstW; ++dx)
                            {
                                int sx = sx_ + dx * strideX;
                                if ((sx >= 0) && (sx < srcW))
                                {
                                    *buf++ = src1[sx];
                                }
                                else
                                {
                                    *buf++ = 0;
                                }
                            }
                        }
                    }
                }
            }
        }

        public static unsafe void Im2col(double* src,
                                 int srcC,
                                 int srcH,
                                 int srcW,
                                 int kernelY,
                                 int kernelX,
                                 int dilationY,
                                 int dilationX,
                                 int strideY,
                                 int strideX,
                                 int padY,
                                 int padX,
                                 int padH,
                                 int padW,
                                 double* buf)
        {
            int dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
            int dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
            for (int sc = 0; sc < srcC; ++sc)
            {
                var scsrcH = sc * srcH;
                for (int ky = 0; ky < kernelY; ++ky)
                {
                    int sy_ = ky * dilationY - padY;
                    for (int kx = 0; kx < kernelX; ++kx)
                    {
                        int sx_ = kx * dilationX - padX;
                        for (int dy = 0; dy < dstH; ++dy)
                        {
                            int sy = sy_ + dy * strideY;
                            if ((sy < 0) || (sy >= srcH))
                            {
                                for (int dx = 0; dx < dstW; ++dx)
                                {
                                    *buf++ = 0;
                                }
                                continue;
                            }
                            var src1 = src + (scsrcH + sy) * srcW;
                            for (int dx = 0; dx < dstW; ++dx)
                            {
                                int sx = sx_ + dx * strideX;
                                if ((sx >= 0) && (sx < srcW))
                                {
                                    *buf++ = src1[sx];
                                }
                                else
                                {
                                    *buf++ = 0;
                                }
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Im2Col-based implementation of two-dimensional convolution.
        /// </summary>
        /// <param name="src">Source data.</param>
        /// <param name="batch">Batch size.</param>
        /// <param name="srcC">Input channels.</param>
        /// <param name="srcH">Input height.</param>
        /// <param name="srcW">Input width.</param>
        /// <param name="kernelY">Kernel height.</param>
        /// <param name="kernelX">Kernel width.</param>
        /// <param name="dilationY">Dilation of the kernel by height.</param>
        /// <param name="dilationX">Dilation of the kernel by width.</param>
        /// <param name="strideY">Stride of the convolution by height.</param>
        /// <param name="strideX">Stride of the convolution by width.</param>
        /// <param name="padY">Zero padding by left side.</param>
        /// <param name="padX">Zero padding by top side.</param>
        /// <param name="padH">Zero padding by right side.</param>
        /// <param name="padW">Zero padding by bottom side.</param>
        /// <param name="group">Convolution groups. If group=srcC=dstC, convolution is depthwise separable.</param>
        /// <param name="weight">Weights (kernels).</param>
        /// <param name="bias">Bias.</param>
        /// <param name="dst">Destination memory.</param>
        /// <param name="dstC">Output channels.</param>
        public static unsafe void Conv2D(float* src,
                                        int batch,
                                        int srcC,
                                        int srcH,
                                        int srcW,
                                        int kernelY,
                                        int kernelX,
                                        int dilationY,
                                        int dilationX,
                                        int strideY,
                                        int strideX,
                                        int padY,
                                        int padX,
                                        int padH,
                                        int padW,
                                        int group,
                                        float* weight,
                                        float* dst,
                                        int dstC,
                                        float* bias = null)
        {
            /// <summary>
            /// Matrix multiplication.
            /// </summary>
            /// <param name="M">A rows.</param>
            /// <param name="N">A columns.</param>
            /// <param name="K">B columns.</param>
            /// <param name="A">Left matrix.</param>
            /// <param name="B">Right matrix.</param>
            /// <param name="C">Result matrix.</param>
            unsafe void mm(int M,
                                  int N,
                                  int K,
                                  float* A,
                                  float* B,
                                  float* C)
            {
                for (int i = 0; i < M; i++)
                {
                    var Cp = C + i * N;
                    var Ap = A + i * K;
                    for (int j = 0; j < N; ++j)
                    {
                        Cp[j] = 0;
                    }
                    for (int k = 0; k < K; ++k)
                    {
                        var a = Ap[k];
                        var Bp = B + k * N;
                        for (int j = 0; j < N; ++j)
                        {
                            Cp[j] += a * Bp[j];
                        }
                    }
                }
            }

            int dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
            int dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
            int M = dstC / group;
            int N = dstH * dstW;
            int K = srcC * kernelY * kernelX / group;
            var buf = (float*)Marshal.AllocCoTaskMem(srcC * kernelY * kernelX * dstH * dstW * sizeof(float));
            for (int b = 0; b < batch; ++b)
            {
                Im2col(src, srcC, srcH, srcW, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, buf);
                for (int g = 0; g < group; ++g)
                {
                    mm(M, N, K, weight + M * K * g, buf + N * K * g, dst + M * N * g);
                }

                if (bias != null)
                {
                    for (int i = 0; i < dstC; ++i)
                    {
                        var pdst = dst + i * N;
                        for (int j = 0; j < N; ++j)
                        {
                            pdst[j] += bias[i];
                        }
                    }
                }
                src += srcC * srcH * srcW;
                dst += dstC * dstH * dstW;
            }
            Marshal.FreeCoTaskMem((IntPtr)buf);
        }


        /// <summary>
        /// Im2Col-based implementation of two-dimensional convolution.
        /// </summary>
        /// <param name="src">Source data.</param>
        /// <param name="batch">Batch size.</param>
        /// <param name="srcC">Input channels.</param>
        /// <param name="srcH">Input height.</param>
        /// <param name="srcW">Input width.</param>
        /// <param name="kernelY">Kernel height.</param>
        /// <param name="kernelX">Kernel width.</param>
        /// <param name="dilationY">Dilation of the kernel by height.</param>
        /// <param name="dilationX">Dilation of the kernel by width.</param>
        /// <param name="strideY">Stride of the convolution by height.</param>
        /// <param name="strideX">Stride of the convolution by width.</param>
        /// <param name="padY">Zero padding by left side.</param>
        /// <param name="padX">Zero padding by top side.</param>
        /// <param name="padH">Zero padding by right side.</param>
        /// <param name="padW">Zero padding by bottom side.</param>
        /// <param name="group">Convolution groups. If group=srcC=dstC, convolution is depthwise separable.</param>
        /// <param name="weight">Weights (kernels).</param>
        /// <param name="bias">Bias.</param>
        /// <param name="dst">Destination memory.</param>
        /// <param name="dstC">Output channels.</param
        public static unsafe void Conv2D(double* src,
                                        int batch,
                                        int srcC,
                                        int srcH,
                                        int srcW,
                                        int kernelY,
                                        int kernelX,
                                        int dilationY,
                                        int dilationX,
                                        int strideY,
                                        int strideX,
                                        int padY,
                                        int padX,
                                        int padH,
                                        int padW,
                                        int group,
                                        double* weight,
                                        double* bias,
                                        double* dst,
                                        int dstC)
        {
            /// <summary>
            /// Matrix multiplication.
            /// </summary>
            /// <param name="M">A rows.</param>
            /// <param name="N">A columns.</param>
            /// <param name="K">B columns.</param>
            /// <param name="A">Left matrix.</param>
            /// <param name="B">Right matrix.</param>
            /// <param name="C">Result matrix.</param>
            unsafe void mm(int M,
                                  int N,
                                  int K,
                                  double* A,
                                  double* B,
                                  double* C)
            {
                for (int i = 0; i < M; i++)
                {
                    var Cp = C + i * N;
                    var Ap = A + i * K;
                    for (int j = 0; j < N; ++j)
                    {
                        Cp[j] = 0;
                    }
                    for (int k = 0; k < K; ++k)
                    {
                        var a = Ap[k];
                        var Bp = B + k * N;
                        for (int j = 0; j < N; ++j)
                        {
                            Cp[j] += a * Bp[j];
                        }
                    }
                };
            }

            int dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
            int dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
            int M = dstC / group;
            int N = dstH * dstW;
            int K = srcC * kernelY * kernelX / group;
            var buf = (double*)Marshal.AllocCoTaskMem(srcC * kernelY * kernelX * dstH * dstW * sizeof(double));
            for (int b = 0; b < batch; ++b)
            {
                Im2col(src, srcC, srcH, srcW, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, buf);
                for (int g = 0; g < group; ++g)
                {
                    mm(M, N, K, weight + M * K * g, buf + N * K * g, dst + M * N * g);
                    //cs_BLAS.DGEMM("nota", "notb", M, N, K, 0.0f, )
                }
                for (int i = 0; i < dstC; ++i)
                {
                    var pdst = dst + i * N;
                    for (int j = 0; j < N; ++j)
                    {
                        pdst[j] += bias[i];
                    }
                }
                src += srcC * srcH * srcW;
                dst += dstC * dstH * dstW;
            }
            Marshal.FreeCoTaskMem((IntPtr)buf);
        }
    }
}
