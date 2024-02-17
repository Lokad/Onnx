using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lokad.Onnx.Tensors
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
            else if (lda < Math.Max(1, nrowa))
            {
                info = 8;
            }
            else if (ldb < Math.Max(1, nrowb))
            {
                info = 10;
            }
            else if (ldc < Math.Max(1, m))
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
}
