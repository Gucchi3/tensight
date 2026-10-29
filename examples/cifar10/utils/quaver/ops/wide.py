# @file wide.py
# @brief Exact rounding of wide products (up to about 80 bits) with int64 only, for wide accumulators and block scales
# @author Kohei Yamaguchi, Claude

import torch

from ..formats import NumberFormat, RoundMode

# a x multiplier is kept as high x 2^WIDE_SPLIT + low, with 0 <= low < 2^WIDE_SPLIT
WIDE_SPLIT = 32
WIDE_LIMIT = 2**61



def wide_round(a: torch.Tensor, multiplier: torch.Tensor, divisor: torch.Tensor, shift: torch.Tensor, mode: RoundMode) -> torch.Tensor:
    '''
    #* Return round(a x multiplier / (divisor x 2^shift)) exactly, with int64 math only (the tensors broadcast).
    #? The product a x multiplier can need up to 82 bits, so it is kept in two parts. Wide accumulators and block scales use this, where float64 would not be exact.
    #! |a| < 2^62, 0 <= multiplier < 2^20, 1 <= divisor < 2^20, and -30 <= shift (a shift over 94 counts as 94). A result of 2^40 or more is not exact: it only stays that large, so clamp it after.
    '''
    
    # The inputs as int64 of one shape (a shift over 94 counts as 94: the result is 0 then)
    a, multiplier, divisor, shift = torch.broadcast_tensors(a.long(), multiplier.long(), divisor.long(), shift.long().clamp(max=94))
    
    # a x multiplier = high x 2^32 + low, with 0 <= low < 2^32 (a = a_high x 2^32 + a_low, and the carry of a_low x multiplier goes to high)
    mask = 2**WIDE_SPLIT - 1
    low  = (a & mask) * multiplier
    high = (a >> WIDE_SPLIT) * multiplier + (low >> WIDE_SPLIT)
    low  = low & mask
    
    # Shifts of 32 or more: the whole part comes from high, and the rest is (rest_high, low) with rest_high < 2^(shift - 32)
    above      = (shift - WIDE_SPLIT).clamp(min=0)
    whole_wide = high >> above
    rest_high  = high - (whole_wide << above)
    
    # Shifts below 32, when |a x multiplier| < 2^61: take the product as one int64, and shift it (a left shift that passes WIDE_LIMIT only stays large)
    ones        = torch.ones_like(shift)
    small       = high.abs() < 2**29
    product     = (torch.where(small, high, 0) << WIDE_SPLIT) + low
    grows       = product.abs() < (WIDE_LIMIT >> (-shift).clamp(min=0, max=62))
    left        = torch.where(grows, product, 0) << (-shift).clamp(min=0)
    low_bits    = (ones << shift.clamp(min=1, max=WIDE_SPLIT - 1)) - 1
    whole_small = torch.where(shift >= 0, product >> shift.clamp(min=0, max=WIDE_SPLIT - 1), torch.where(grows, left, torch.where(product < 0, -WIDE_LIMIT, WIDE_LIMIT)))
    rest_small  = torch.where(shift > 0 , product & low_bits                               , 0                                                                          )
    
    # Shifts below 32, when |a x multiplier| >= 2^61: high moves up (past WIDE_LIMIT the result only stays large), and low gives the rest
    below      = (WIDE_SPLIT - shift).clamp(min=0, max=62)
    safe       = high.abs() < (WIDE_LIMIT >> below)
    moved      = (torch.where(safe, high, 0) << below) + (low >> shift.clamp(min=0, max=WIDE_SPLIT - 1))
    whole_big  = torch.where(safe     , moved         , torch.where(high < 0, -WIDE_LIMIT, WIDE_LIMIT))
    rest_big   = torch.where(shift > 0, low & low_bits, 0                                             )
    whole_near = torch.where(small    , whole_small   , whole_big                                     )
    rest_near  = torch.where(small    , rest_small    , rest_big                                      )
    
    # The whole part over 2^shift, then the division by the divisor: value = quotient + (remainder x 2^shift + rest) / (divisor x 2^shift)
    wide      = shift >= WIDE_SPLIT
    whole     = torch.where(wide, whole_wide, whole_near)
    quotient  = torch.div(whole, divisor, rounding_mode="floor")
    remainder = whole - quotient * divisor
    
    # Compare the rest with half of 2^shift: -1 below, 0 equal, 1 above (a rest of (rest_high, low) is rest_high x 2^32 + low)
    half_high   = torch.where(above > 0             , ones << (above - 1).clamp(min=0)                                              , 0                         )
    half_low    = torch.where(above > 0             , 0                                                                             , 2 ** (WIDE_SPLIT - 1)     )
    compare_far = torch.where(rest_high != half_high, torch.sign(rest_high - half_high)                                             , torch.sign(low - half_low))
    compare_mid = torch.where(shift > 0             , torch.sign(rest_near - (ones << (shift - 1).clamp(min=0, max=WIDE_SPLIT - 1))), -1                        )
    compare     = torch.where(wide                  , compare_far                                                                   , compare_mid               )
    rest        = torch.where(wide                  , (rest_high != 0) | (low != 0)                                                 , rest_near != 0            )
    
    # Twice the fraction against 1: (2 x remainder - divisor) x 2^shift + 2 x rest. Above, a tie, or below
    double  = 2 * remainder - divisor
    greater = (double >= 1) | ((double == -1) & (compare > 0)) | ((double == 0) & rest)
    tie     = ((double == -1) & (compare == 0)) | ((double == 0) & ~rest)
    
    # Round with the mode (floor never goes up)
    if mode == "floor":
        return quotient
    tie_up = {"half_up": torch.ones_like(tie), "half_even": quotient % 2 == 1, "half_away": a >= 0}[mode]
    return quotient + (greater | (tie & tie_up)).long()


def wide_round_format(a: torch.Tensor, multiplier: torch.Tensor, divisor: torch.Tensor, shift: torch.Tensor, number_format: NumberFormat, mode: RoundMode, zero_point: torch.Tensor | None = None) -> torch.Tensor:
    '''
    #* Round a x multiplier / (divisor x 2^shift), a value in code units, to a code of number_format exactly, add the zero point, and stop at the ends of the range.
    #? INT formats round with mode. FP formats round half to even on their grid (the step doubles with each power of two), on the size and then with the sign, like format_quantize().
    #! The same limits as wide_round(). The FP grid needs the size, so it uses floor first to find the power of two.
    '''
    
    # INT formats: round, add the zero point, and clamp
    low  = round(number_format.min_value / number_format.step_min)
    high = round(number_format.max_value / number_format.step_min)
    zero = torch.zeros_like(a) if zero_point is None else zero_point.long()
    if number_format.kind == "int":
        return (wide_round(a, multiplier, divisor, shift, mode).clamp(-WIDE_LIMIT, WIDE_LIMIT) + zero).clamp(low, high)
    
    # FP formats: the power of two of the size (codes at or above the top only clamp), and the step there (1 up to 2^(mantissa + 1), then doubling)
    size  = a.abs()
    whole = wide_round(size, multiplier, divisor, shift, "floor").clamp(0, high)
    power = torch.frexp(whole.double())[1].long() - 1
    steps = (power - number_format.mantissa_bits).clamp(min=0)
    
    # Round half to even with that step, then put the sign back and clamp
    codes = wide_round(size, multiplier, divisor, shift + steps, "half_even").clamp(0, high) << steps
    return torch.where(a < 0, -codes, codes).clamp(low, high)
