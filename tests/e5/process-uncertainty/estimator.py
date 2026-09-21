"""Paired Fieller intervals across processes; never across calls in one process.

The t inversion assumes independent cohorts and a sufficiently normal paired
contrast. It does not establish those assumptions from timing data. Exact
rational moments avoid cancellation for proportional or nearly equal arrays.
"""
from fractions import Fraction
import math


def _cos_integral(angle, power):
    """Integral of cos(x)**power from zero to angle, for nonnegative integers."""
    sine, cosine = math.sin(angle), math.cos(angle)
    even, odd = angle, sine
    for exponent in range(2, power + 1):
        if exponent % 2 == 0:
            even = sine * cosine ** (exponent - 1) / exponent + (exponent - 1) * even / exponent
        else:
            odd = sine * cosine ** (exponent - 1) / exponent + (exponent - 1) * odd / exponent
    return odd if power % 2 else even


def student_critical(confidence, df):
    """Positive Student t quantile with P(-t <= T_df <= t) = confidence.

    Substitution t=sqrt(df)*tan(angle) reduces the central probability to a
    normalized integral of cos(angle)**(df-1). The integer-power recurrence
    avoids a numerical-library dependency and is independently tested against
    closed forms for df=1,2 and numerical quadrature for other degrees.
    """
    if type(df) is not int or not 1 <= df <= 10000:
        raise ValueError("positive integer degrees of freedom required (at most10000)")
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    low, high = 0.0, math.pi / 2
    normalizer = _cos_integral(high, df - 1)
    for _ in range(80):
        angle = (low + high) / 2
        if _cos_integral(angle, df - 1) / normalizer < confidence:
            low = angle
        else:
            high = angle
    return math.sqrt(df) * math.tan((low + high) / 2)


def _values(values):
    result = list(values)
    if len(result) < 3:
        raise ValueError("at least three independent process pairs required")
    if any(type(v) not in (int, Fraction) or v <= 0 for v in result):
        raise ValueError("process means must be positive integers or exact Fractions")
    return [Fraction(v) for v in result]


def fieller(numerator, denominator, critical):
    """Invert the paired t test for the ratio of arithmetic process means.

    Each array has one mean per fresh process, in matching cohort order.
    No tail is removed. A nonpositive quadratic leading coefficient is reported
    as unbounded, never as a bounded interval or a passing result.
    """
    y, x = _values(numerator), _values(denominator)
    if len(y) != len(x):
        raise ValueError("paired process counts differ")
    if type(critical) not in (int, float) or not math.isfinite(critical) or critical <= 0:
        raise ValueError("finite positive critical value required")
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    vx = sum((v - mx) ** 2 for v in x) / (n * (n - 1))
    vy = sum((v - my) ** 2 for v in y) / (n * (n - 1))
    covariance = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / (n * (n - 1))
    q = Fraction(critical) ** 2
    # Divide all coefficients by mx**2, retaining exact cancellation.
    a = 1 - q * vx / mx ** 2
    b = my / mx - q * covariance / mx ** 2
    c = (my ** 2 - q * vy) / mx ** 2
    discriminant = b ** 2 - a * c
    result = dict(n=n, ratio=float(my / mx), numerator_mean=float(my), denominator_mean=float(mx),
                  numerator_mean_variance=float(vy), denominator_mean_variance=float(vx),
                  mean_covariance=float(covariance), critical=float(critical),
                  coefficients=[float(a), float(-2 * b), float(c)],
                  bounded=False, interval=None, reason=None)
    if a <= 0:
        result['reason'] = 'denominator uncertainty yields an unbounded confidence set'
    elif discriminant < 0:
        # Positive observations give a nonempty set containing my/mx. Refuse
        # impossible arithmetic instead of clipping a negative discriminant.
        raise ArithmeticError("negative Fieller discriminant")
    else:
        center, radius = float(b / a), math.sqrt(float(discriminant / a ** 2))
        result.update(bounded=True, interval=[center - radius, center + radius])
    return result


def process_mean(rows, boundary, frequency):
    """Reduce all measured integer ticks to ONE exact process mean in seconds."""
    if boundary not in ('execute', 'request'):
        raise ValueError("unknown timing boundary")
    if type(frequency) is not int or frequency <= 0:
        raise ValueError("positive integer clock frequency required")
    values = list(rows)
    if not values:
        raise ValueError("empty measured process")
    for row in values:
        if not isinstance(row, dict) or any(type(row.get(k)) is not int or row[k] <= 0 for k in ('execute', 'request')):
            raise ValueError("positive integer duration ticks required")
        if row['request'] < row['execute']:
            raise ValueError("enclosing request is shorter than Execute")
    return Fraction(sum(row[boundary] for row in values), len(values) * frequency)


def contained(result, low, high):
    """Unbounded sets can never qualify a bounded performance assertion."""
    return result['bounded'] and low <= result['interval'][0] <= result['interval'][1] <= high
