using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Lokad.Onnx
{

    /// <summary>
    /// Brain floating-point value: one sign bit, eight exponent bits biased like
    /// float32, and seven mantissa bits. Every value widens to float32 exactly;
    /// narrowing rounds to nearest with ties to even, and NaN canonicalizes.
    /// Implemented from the format definition; no external source.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly struct BFloat16 :
        IComparable,
        IComparable<BFloat16>,
        IEquatable<BFloat16>
    {
        internal const ushort SignMask = 0x8000;
        internal const ushort BiasedExponentMask = 0x7F80;
        internal const int BiasedExponentShift = 7;
        internal const ushort TrailingSignificandMask = 0x007F;
        internal const byte MaxBiasedExponent = 0xFF;

        // Constants representing the private bit-representation for various default values

        private const ushort PositiveZeroBits = 0x0000;
        private const ushort NegativeZeroBits = 0x8000;

        private const ushort OneBits = 0x3F80;  // 0b0_01111111_0000000

        private const ushort PositiveInfinityBits = 0x7F80;
        private const ushort NegativeInfinityBits = 0xFF80;

        private const ushort PositiveQNaNBits = 0x7FC1;
        private const ushort NegativeQNaNBits = 0xFFC1;

        private const ushort MinValueBits = 0xFF7F; // 1b0_11111110_1111111
        private const ushort MaxValueBits = 0x7F7F; // 0b0_11111110_1111111

        private const ushort EpsilonBits = 0x0080; // the smallest positive normal value

        private const ushort PiBits = 0x4049; // 0b0_10000000_1001001

        // Used for rounding subnormal values
        private const uint RoundingBase = 0x7FFF;

        // Well-defined and commonly used values

        /// <summary>
        /// BFloat16 Epsilon value
        /// </summary>
        public static BFloat16 Epsilon => new BFloat16(EpsilonBits);

        /// <summary>
        /// BFloat16 Pi value
        /// </summary>
        public static BFloat16 Pi => new BFloat16(PiBits);

        /// <summary>
        /// BFloat16 Positive infinity value
        /// </summary>
        public static BFloat16 PositiveInfinity => new BFloat16(PositiveInfinityBits);

        /// <summary>
        /// BFloat16 Negative infinity value
        /// </summary>
        public static BFloat16 NegativeInfinity => new BFloat16(NegativeInfinityBits);

        /// <summary>
        /// BFloat16 NaN
        /// </summary>
        public static BFloat16 NaN => new BFloat16(NegativeQNaNBits);

        /// <summary>
        /// BFloat16 Positive Zero
        /// </summary>
        public static BFloat16 Zero => new BFloat16(PositiveZeroBits);  // 0.0

        /// <summary>
        /// BFloat16 One
        /// </summary>
        public static BFloat16 One => new BFloat16(OneBits);  // 1.0

        /// <summary>
        /// BFloat16 Negative Zero
        /// </summary>
        public static BFloat16 NegativeZero => new BFloat16(NegativeZeroBits);  // -0.0

        /// <summary>
        /// BFloat16 Min value
        /// </summary>
        public static BFloat16 MinValue => new BFloat16(MinValueBits);  // 65,407

        /// <summary>
        /// BFloat16 Max value
        /// </summary>

        public static BFloat16 MaxValue => new BFloat16(MaxValueBits); // 32,639

        /// <summary>
        /// bfloat16 representation bits
        /// </summary>
        public readonly ushort value;

        /// <summary>
        /// Constructor from ushort, no conversion takes place. The value
        /// is assumed to be converted
        /// </summary>
        /// <param name="v">bfloat16 representation bits</param>
        public BFloat16(ushort v)
        {
            value = v;
        }

        static bool HasSign(BFloat16 v) => (v.value & SignMask) != 0;

        static int Exponent(BFloat16 v) => (v.value & BiasedExponentMask) >> BiasedExponentShift;

        static uint Mantissa(BFloat16 v) => (uint)(v.value & TrailingSignificandMask);

        static bool IsZero(BFloat16 v) => (v.value & ~SignMask) == 0;

        /// <summary>
        /// Compares two BFloat16 instances.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns>true if the left is less than right according to IEEE</returns>
        public static bool operator <(BFloat16 left, BFloat16 right)
        {
            // NaN is unordered with respect to everything, including itself.
            if (IsNaN(left) || IsNaN(right)) return false;
            // Exact widening preserves order, including signed zero.
            return (float)left < (float)right;
        }

        /// <summary>
        /// Compares two BFloat16 instances.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns>true if the left is greater than right according to IEEE</returns>
        public static bool operator >(BFloat16 left, BFloat16 right)
        {
            return right < left;
        }

        /// <summary>
        /// Compares two BFloat16 instances.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns>true if the left is less or equal than right according to IEEE</returns>
        public static bool operator <=(BFloat16 left, BFloat16 right)
        {
            if (IsNaN(left) || IsNaN(right)) return false;
            return (float)left <= (float)right;
        }

        /// <summary>
        /// Compares two BFloat16 instances.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns>true if the left is greater or equal than right according to IEEE</returns>
        public static bool operator >=(BFloat16 left, BFloat16 right)
        {
            return right <= left;
        }

        /// <summary>
        /// Binary equality: same bits, except NaN never equals, even itself.
        /// Note signed zeros differ here, unlike ordered comparison.
        /// </summary>
        public static bool operator ==(BFloat16 left, BFloat16 right)
        {
            if (IsNaN(left) || IsNaN(right)) return false;
            return left.value == right.value;
        }

        /// <summary>Negation of binary equality.</summary>
        public static bool operator !=(BFloat16 left, BFloat16 right)
        {
            return !(left == right);
        }

        /// <summary>
        /// Determines whether the specified value is finite (zero, subnormal, or normal).
        /// </summary>
        /// <param name="value">BFloat16 instance.</param>
        /// <returns>true if the value is finite</returns>
        public static bool IsFinite(BFloat16 value)
        {
            return Exponent(value) != MaxBiasedExponent;
        }

        /// <summary>
        /// Determines whether the specified value is infinite.
        /// </summary>
        /// <param name="value">BFloat16 instance.</param>
        /// <returns>true if the value is infinite</returns>
        public static bool IsInfinity(BFloat16 value)
        {
            return Exponent(value) == MaxBiasedExponent && Mantissa(value) == 0;
        }

        /// <summary>
        /// Determines whether the specified value is NaN.
        /// </summary>
        /// 
        /// <param name="value">BFloat16 instance</param>
        /// <returns>true if the value is not a number</returns>
        public static bool IsNaN(BFloat16 value)
        {
            return Exponent(value) == MaxBiasedExponent && Mantissa(value) != 0;
        }

        /// <summary>
        /// Determines whether the specified value is negative.
        /// </summary>
        /// <param name="value">BFloat16 instance</param>
        /// <returns>true if the value is negative</returns></returns>
        public static bool IsNegative(BFloat16 value)
        {
            return HasSign(value);
        }

        /// <summary>
        /// Determines whether the specified value is negative infinity.
        /// </summary>
        /// 
        /// <param name="value">BFloat16 instance</param>
        /// <returns>true if the value is negative infinity</returns>
        public static bool IsNegativeInfinity(BFloat16 value)
        {
            return value.value == NegativeInfinityBits;
        }

        /// <summary>
        /// Determines whether the specified value is normal
        /// </summary>
        /// <param name="value"></param>
        /// <returns>true or false</returns>
        public static bool IsNormal(BFloat16 value)
        {
            int exponent = Exponent(value);
            return exponent != 0 && exponent != MaxBiasedExponent;
        }

        /// <summary>
        /// Determines whether the specified value is positive infinity.
        /// </summary>
        /// <param name="value">BFloat16 instance</param>
        /// <returns></returns>
        public static bool IsPositiveInfinity(BFloat16 value)
        {
            return value.value == PositiveInfinityBits;
        }

        /// <summary>
        /// Determines whether the specified value is subnormal.
        /// </summary>
        /// <param name="value">BFloat16 instance</param>
        /// <returns>true if the value is subnormal</returns>
        public static bool IsSubnormal(BFloat16 value)
        {
            return Exponent(value) == 0 && Mantissa(value) != 0;
        }

        /// <summary>
        /// Compares this object to another object, returning an integer that indicates the relationship.
        /// </summary>
        /// 
        /// <param name="obj">Object to compare to</param>
        /// <returns>A value less than zero if this is less than <paramref name="obj"/>,
        /// zero if this is equal to <paramref name="obj"/>, or a value greater than zero
        /// if this is greater than <paramref name="obj"/>.
        /// </returns>
        /// <exception cref="ArgumentException">Thrown when <paramref name="obj"/> is not of type <see cref="BFloat16"/>.</exception>
        public int CompareTo(object? obj)
        {
            if (!(obj is BFloat16))
            {
                return (obj is null) ? 1 : throw new ArgumentException("Object must be of type BFloat16");
            }
            return CompareTo((BFloat16)(obj));
        }

        /// <summary>
        /// Compares this object to another object, returning an integer that indicates the relationship.
        /// </summary>
        /// <param name="other">Object to compare to</param>
        /// <returns>A value less than zero if this is less than <paramref name="other"/>,
        /// zero if this is equal to <paramref name="other"/>, 
        /// or a value greater than zero if this is greater than <paramref name="other"/>.</returns>
        public int CompareTo(BFloat16 other)
        {
            if (this < other)
            {
                return -1;
            }

            if (this > other)
            {
                return 1;
            }

            if (this == other)
            {
                return 0;
            }

            if (IsNaN(this))
            {
                return IsNaN(other) ? 0 : -1;
            }

            Debug.Assert(IsNaN(other));
            return 1;
        }

        /// <summary>
        /// Returns a value indicating whether this instance and other BFloat16 represent the same value.
        /// </summary>
        /// <param name="other">A BFloat16 object to compare to this instance.</param>
        /// <returns>true if other.value is equal to this instance; otherwise, false.</returns>
        public bool Equals(BFloat16 other)
        {
            return value == other.value
                || (IsZero(this) && IsZero(other))
                || (IsNaN(this) && IsNaN(other));
        }

        /// <summary>
        /// Returns a value indicating whether this instance and a specified System.Object
        /// represent the same type and value.
        /// </summary>
        /// <param name="obj">An System.Object.</param>
        /// <returns>true if obj is BFloat16 its value is equal to this instance; otherwise, false.</returns>
        public override bool Equals(object? obj)
        {
            return (obj is BFloat16 other) && Equals(other);
        }

        /// <summary>
        /// Returns the hash code for this instance.
        /// </summary>
        /// <returns>A 32-bit signed integer hash code.</returns>
        public override int GetHashCode()
        {
            if (IsNaN(this) || IsZero(this))
            {
                // All NaNs share a hash code, as do both zeros, matching Equals.
                return value & PositiveInfinityBits;
            }
            return value;
        }

        /// <summary>
        /// Returns a string representation of the current value.
        /// </summary>
        /// <returns>Text representation of BFloat16</returns>
        public override string ToString()
        {
            return $"{value} : {ToFloat()}";
        }

        /// <summary>
        /// Explicit conversion
        /// </summary>
        /// <returns>single precision value converted from BFloat16</returns>
        public float ToFloat()
        {
            return (float)this;
        }

        /// <summary>Explicitly converts a <see cref="float" /> value to its nearest representable bfloat16 value.</summary>
        /// <param name="value">The value to convert.</param>
        /// <returns><paramref name="value" /> converted to its nearest representable half-precision floating-point value.</returns>
        public static explicit operator BFloat16(float value)
        {
            // NaN canonicalizes: payloads do not survive narrowing.
            if (float.IsNaN(value)) return NaN;
            // Round to nearest, ties to even: bias by half a unit plus the kept
            // lowest bit, then truncate. Infinities and overflow to infinity
            // fall out of the same addition with no special case.
            uint bits = BitConverter.SingleToUInt32Bits(value);
            uint rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
            return new BFloat16((ushort)(rounded >> 16));
        }

        /// <summary>
        /// Explicitly converts a BFloat16 value to its nearest representable <see cref="float" /> value.
        /// </summary>
        /// <param name="value">The value to convert.</param>
        /// <returns><paramref name="value" /> converted to its nearest representable <see cref="float" /> value.</returns>
        public static explicit operator float(BFloat16 value)
        {
            uint bits = value.value;
            bool sign = (bits & SignMask) != 0;
            uint exponent = (bits & BiasedExponentMask) >> BiasedExponentShift;
            uint mantissa = bits & TrailingSignificandMask;
            if (exponent == MaxBiasedExponent)
            {
                if (mantissa == 0) return sign ? float.NegativeInfinity : float.PositiveInfinity;
                // Quiet bit set plus the payload shifted into place.
                return BitConverter.UInt32BitsToSingle(((sign ? 1u : 0u) << 31) | 0x7FC00000u | (mantissa << 15));
            }
            if (bits == PositiveZeroBits || bits == NegativeZeroBits) return sign ? -0.0f : 0.0f;
            // Shared exponent bias makes widening an exact left shift.
            if (!BitConverter.IsLittleEndian) return BitConverter.UInt32BitsToSingle(bits);
            return BitConverter.UInt32BitsToSingle(bits << 16);
        }
    }
}