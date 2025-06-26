"""Unit tests for crop_utils module."""

import pytest

from personfromvid.output.crop_utils import (
    calculate_fixed_aspect_ratio_bbox,
    normalize_bbox,
    parse_aspect_ratio,
    validate_bbox,
)


class TestParseAspectRatio:
    """Tests for parse_aspect_ratio function."""

    def test_valid_standard_ratios(self):
        """Test parsing of common valid aspect ratios."""
        # Standard ratios
        assert parse_aspect_ratio("16:9") == (16, 9)
        assert parse_aspect_ratio("4:3") == (4, 3)
        assert parse_aspect_ratio("1:1") == (1, 1)
        assert parse_aspect_ratio("21:9") == (21, 9)
        assert parse_aspect_ratio("3:2") == (3, 2)
        assert parse_aspect_ratio("9:16") == (9, 16)  # Vertical

    def test_valid_unusual_ratios(self):
        """Test parsing of valid but unusual aspect ratios."""
        # Extreme ratios within bounds
        assert parse_aspect_ratio("100:1") == (100, 1)  # Ratio = 100.0 (max)
        assert parse_aspect_ratio("1:10") == (1, 10)    # Ratio = 0.1 (min)
        assert parse_aspect_ratio("2:20") == (2, 20)    # Ratio = 0.1 (min)
        assert parse_aspect_ratio("50:1") == (50, 1)    # Ratio = 50.0

        # Large numbers
        assert parse_aspect_ratio("1920:1080") == (1920, 1080)
        assert parse_aspect_ratio("3840:2160") == (3840, 2160)

    def test_valid_ratio_bounds(self):
        """Test aspect ratios at the boundary limits."""
        # Test exactly at bounds
        assert parse_aspect_ratio("1:10") == (1, 10)    # Ratio = 0.1 (exactly min)
        assert parse_aspect_ratio("100:1") == (100, 1)  # Ratio = 100.0 (exactly max)

        # Test just inside bounds
        assert parse_aspect_ratio("11:100") == (11, 100)  # Ratio = 0.11 (just above min)
        assert parse_aspect_ratio("99:1") == (99, 1)      # Ratio = 99.0 (just below max)

    def test_invalid_format_strings(self):
        """Test rejection of malformed aspect ratio strings."""
        # Missing parts
        assert parse_aspect_ratio("16:") is None
        assert parse_aspect_ratio(":9") is None
        assert parse_aspect_ratio(":") is None
        assert parse_aspect_ratio("") is None

        # Wrong separators
        assert parse_aspect_ratio("16/9") is None
        assert parse_aspect_ratio("16-9") is None
        assert parse_aspect_ratio("16x9") is None
        assert parse_aspect_ratio("16*9") is None
        assert parse_aspect_ratio("16.9") is None

        # Multiple separators
        assert parse_aspect_ratio("16:9:1") is None
        assert parse_aspect_ratio("16::9") is None

        # Invalid characters
        assert parse_aspect_ratio("16:9a") is None
        assert parse_aspect_ratio("a16:9") is None
        assert parse_aspect_ratio("16a:9") is None
        assert parse_aspect_ratio("16:9b") is None
        assert parse_aspect_ratio("1.6:9") is None
        assert parse_aspect_ratio("16:9.0") is None

    def test_invalid_decimal_ratios(self):
        """Test rejection of decimal aspect ratios as specified."""
        # Decimal ratios (should be rejected as per spec)
        assert parse_aspect_ratio("1.78:1") is None
        assert parse_aspect_ratio("1:1.33") is None
        assert parse_aspect_ratio("1.5:1.0") is None
        assert parse_aspect_ratio("16.0:9.0") is None

    def test_invalid_zero_and_negative_values(self):
        """Test rejection of zero and negative values."""
        # Zero values
        assert parse_aspect_ratio("0:9") is None
        assert parse_aspect_ratio("16:0") is None
        assert parse_aspect_ratio("0:0") is None

        # Negative values
        assert parse_aspect_ratio("-16:9") is None
        assert parse_aspect_ratio("16:-9") is None
        assert parse_aspect_ratio("-16:-9") is None

    def test_invalid_ratio_bounds(self):
        """Test rejection of ratios outside the 0.1-100.0 range."""
        # Ratios too small (< 0.1)
        assert parse_aspect_ratio("1:11") is None     # Ratio = 0.090909... < 0.1
        assert parse_aspect_ratio("1:100") is None    # Ratio = 0.01 < 0.1
        assert parse_aspect_ratio("1:1000") is None   # Ratio = 0.001 < 0.1
        assert parse_aspect_ratio("9:100") is None    # Ratio = 0.09 < 0.1

        # Ratios too large (> 100.0)
        assert parse_aspect_ratio("101:1") is None    # Ratio = 101.0 > 100.0
        assert parse_aspect_ratio("1000:1") is None   # Ratio = 1000.0 > 100.0
        assert parse_aspect_ratio("200:1") is None    # Ratio = 200.0 > 100.0

    def test_invalid_input_types(self):
        """Test rejection of non-string input types."""
        # Non-string types
        assert parse_aspect_ratio(None) is None
        assert parse_aspect_ratio(16) is None
        assert parse_aspect_ratio(16.9) is None
        assert parse_aspect_ratio([16, 9]) is None
        assert parse_aspect_ratio((16, 9)) is None
        assert parse_aspect_ratio({'width': 16, 'height': 9}) is None

    def test_whitespace_handling(self):
        """Test handling of whitespace in input strings."""
        # Leading/trailing whitespace should be stripped
        assert parse_aspect_ratio(" 16:9 ") == (16, 9)
        assert parse_aspect_ratio("\t16:9\t") == (16, 9)
        assert parse_aspect_ratio("\n16:9\n") == (16, 9)

        # Internal whitespace should be rejected
        assert parse_aspect_ratio("16 : 9") is None
        assert parse_aspect_ratio("16: 9") is None
        assert parse_aspect_ratio("16 :9") is None

    def test_edge_case_strings(self):
        """Test edge case string inputs."""
        # Random strings
        assert parse_aspect_ratio("invalid") is None
        assert parse_aspect_ratio("aspect_ratio") is None
        assert parse_aspect_ratio("16by9") is None
        assert parse_aspect_ratio("widescreen") is None
        assert parse_aspect_ratio("square") is None

        # Numbers without separator
        assert parse_aspect_ratio("169") is None
        assert parse_aspect_ratio("43") is None

        # Unicode and special characters
        assert parse_aspect_ratio("16：9") is None  # Full-width colon
        assert parse_aspect_ratio("16∶9") is None   # Ratio symbol
        assert parse_aspect_ratio("16÷9") is None   # Division symbol

    def test_function_documentation_examples(self):
        """Test examples from function docstring."""
        # Examples from docstring should work
        assert parse_aspect_ratio("16:9") == (16, 9)
        assert parse_aspect_ratio("1:1") == (1, 1)
        assert parse_aspect_ratio("invalid") is None

    def test_specification_compliance(self):
        """Test compliance with specification requirements."""
        # Specification: "standard WxH aspect ratio format with positive integers only"
        # Specification: "calculated ratio (width/height) must be between 0.1 and 100.0 inclusive"
        # Specification: "Malformed strings like '16:', ':9', '16/9', or decimal ratios like '1.78:1' must be rejected"

        # Valid per specification
        assert parse_aspect_ratio("16:9") == (16, 9)
        assert parse_aspect_ratio("4:3") == (4, 3)
        assert parse_aspect_ratio("1:1") == (1, 1)

        # Invalid per specification
        assert parse_aspect_ratio("16:") is None
        assert parse_aspect_ratio(":9") is None
        assert parse_aspect_ratio("16/9") is None
        assert parse_aspect_ratio("1.78:1") is None

        # Boundary compliance
        assert parse_aspect_ratio("1:10") == (1, 10)    # 0.1 ratio (valid)
        assert parse_aspect_ratio("100:1") == (100, 1)  # 100.0 ratio (valid)
        assert parse_aspect_ratio("1:11") is None        # < 0.1 ratio (invalid)
        assert parse_aspect_ratio("101:1") is None       # > 100.0 ratio (invalid)


class TestCalculateFixedAspectRatioBbox:
    """Tests for calculate_fixed_aspect_ratio_bbox function."""

    def test_center_bbox_square_ratio(self):
        """Test square aspect ratio with centered bbox."""
        # 100x50 bbox -> 1:1 ratio should be ~71x71 (preserving area)
        result = calculate_fixed_aspect_ratio_bbox((100, 100, 200, 150), (800, 600), (1, 1))
        x1, y1, x2, y2 = result

        # Check aspect ratio is 1:1 (square)
        width = x2 - x1
        height = y2 - y1
        assert width == height, f"Expected square but got {width}x{height}"

        # Check center is approximately preserved
        orig_center_x = (100 + 200) // 2  # 150
        orig_center_y = (100 + 150) // 2  # 125
        new_center_x = (x1 + x2) // 2
        new_center_y = (y1 + y2) // 2

        # Allow some tolerance for integer rounding
        assert abs(new_center_x - orig_center_x) <= 2, f"Center X shifted too much: {new_center_x} vs {orig_center_x}"
        assert abs(new_center_y - orig_center_y) <= 2, f"Center Y shifted too much: {new_center_y} vs {orig_center_y}"

    def test_center_bbox_widescreen_ratio(self):
        """Test widescreen 16:9 aspect ratio with centered bbox."""
        result = calculate_fixed_aspect_ratio_bbox((100, 100, 200, 150), (800, 600), (16, 9))
        x1, y1, x2, y2 = result

        # Check aspect ratio is approximately 16:9
        width = x2 - x1
        height = y2 - y1
        actual_ratio = width / height
        expected_ratio = 16 / 9

        # Allow small tolerance for integer rounding
        assert abs(actual_ratio - expected_ratio) < 0.1, f"Ratio {actual_ratio:.3f} not close to {expected_ratio:.3f}"

    def test_edge_bbox_boundary_shifting(self):
        """Test bbox near image edge with new algorithm that may extend beyond bounds."""
        # Bbox near left edge
        result = calculate_fixed_aspect_ratio_bbox((10, 10, 50, 30), (800, 600), (16, 9))
        x1, y1, x2, y2 = result

        # Check containment of original bbox
        assert x1 <= 10 and x2 >= 50, f"Should contain original X range 10-50, got {x1}-{x2}"
        assert y1 <= 10 and y2 >= 30, f"Should contain original Y range 10-30, got {y1}-{y2}"

        # Check aspect ratio is preserved
        width = x2 - x1
        height = y2 - y1
        actual_ratio = width / height
        expected_ratio = 16 / 9
        assert abs(actual_ratio - expected_ratio) < 0.1, f"Ratio {actual_ratio:.3f} not close to {expected_ratio:.3f}"

    def test_corner_bbox_boundary_handling(self):
        """Test bbox in corner of image with new algorithm."""
        # Top-left corner
        result = calculate_fixed_aspect_ratio_bbox((5, 5, 25, 15), (800, 600), (4, 3))
        x1, y1, x2, y2 = result

        # Check containment of original bbox
        assert x1 <= 5 and x2 >= 25, f"Should contain original X range 5-25, got {x1}-{x2}"
        assert y1 <= 5 and y2 >= 15, f"Should contain original Y range 5-15, got {y1}-{y2}"

        # Check aspect ratio is preserved
        width = x2 - x1
        height = y2 - y1
        actual_ratio = width / height
        expected_ratio = 4 / 3
        assert abs(actual_ratio - expected_ratio) < 0.1, f"Ratio {actual_ratio:.3f} not close to {expected_ratio:.3f}"

    def test_large_bbox_expansion_containment(self):
        """Test that large bbox is expanded to ensure containment with fixed aspect ratio."""
        # Large bbox (700x500) that needs to be made square while ensuring containment
        result = calculate_fixed_aspect_ratio_bbox((50, 50, 750, 550), (800, 600), (1, 1))
        x1, y1, x2, y2 = result

        # Should contain the original bbox (700x500)
        width = x2 - x1
        height = y2 - y1
        original_width = 750 - 50  # 700
        original_height = 550 - 50  # 500
        assert width >= original_width, f"New width {width} should contain original width {original_width}"
        assert height >= original_height, f"New height {height} should contain original height {original_height}"

        # Should be square (1:1 ratio) - the new implementation prioritizes exact aspect ratio
        assert width == height, f"Should be square but got {width}x{height}"

        # For containment of 700x500, we need at least 700x700 square
        assert width >= 700 and height >= 700, f"Should be at least 700x700 to contain source, got {width}x{height}"

    def test_small_bbox_expansion(self):
        """Test that small bbox gets expanded appropriately while ensuring containment."""
        # Very small bbox (5x2)
        result = calculate_fixed_aspect_ratio_bbox((100, 100, 105, 102), (800, 600), (16, 9))
        x1, y1, x2, y2 = result

        # Should contain the original 5x2 bbox
        width = x2 - x1
        height = y2 - y1
        original_width = 105 - 100  # 5
        original_height = 102 - 100  # 2

        assert width >= original_width, f"New width {width} should contain original width {original_width}"
        assert height >= original_height, f"New height {height} should contain original height {original_height}"

        # Should attempt to maintain 16:9 aspect ratio while ensuring containment
        # For such small dimensions, integer rounding affects the ratio significantly
        actual_ratio = width / height

        # With containment-first, the ratio might be different due to minimum size requirements
        # But it should be reasonably close or wider (to contain the 5-pixel width)
        assert actual_ratio >= 1.0, f"Ratio {actual_ratio} should be >= 1.0 for 16:9 target with wide original"

    def test_extreme_aspect_ratios(self):
        """Test handling of extreme aspect ratios with containment guarantee."""
        # Very wide ratio (100:1) - original is 100x50
        result = calculate_fixed_aspect_ratio_bbox((100, 100, 200, 150), (800, 600), (100, 1))
        x1, y1, x2, y2 = result

        width = x2 - x1
        height = y2 - y1

        # Should contain the original 100x50 bbox
        assert width >= 100, f"Width {width} should contain original width 100"
        assert height >= 50, f"Height {height} should contain original height 50"

        # For 100:1 ratio, width should be much larger than height
        # With containment-first, need at least 100 width and resulting height from ratio
        expected_height = max(50, int(100 / 100))  # max(original_height, width/ratio)
        expected_width = max(100, int(50 * 100))   # max(original_width, height*ratio)

        assert width >= expected_width, f"Expected width >= {expected_width} but got {width}"
        assert height >= expected_height, f"Expected height >= {expected_height} but got {height}"

        # Very tall ratio (1:100) - original is 100x50
        result = calculate_fixed_aspect_ratio_bbox((100, 100, 200, 150), (800, 600), (1, 100))
        x1, y1, x2, y2 = result

        width = x2 - x1
        height = y2 - y1

        # Should contain the original 100x50 bbox
        assert width >= 100, f"Width {width} should contain original width 100"
        assert height >= 50, f"Height {height} should contain original height 50"

        # For 1:100 ratio, height should be much larger than width
        # With containment-first, need at least 50 height and resulting width from ratio
        expected_height = max(50, int(100 * 100))  # max(original_height, width*ratio)
        expected_width = max(100, int(50 / 100))   # max(original_width, height/ratio)

        assert width >= expected_width, f"Expected width >= {expected_width} but got {width}"
        assert height >= expected_height, f"Expected height >= {expected_height} but got {height}"

    def test_input_validation(self):
        """Test input validation and error handling."""
        # Invalid bbox (x2 <= x1)
        with pytest.raises(ValueError, match="Invalid bbox"):
            calculate_fixed_aspect_ratio_bbox((100, 100, 100, 150), (800, 600), (1, 1))

        # Invalid bbox (y2 <= y1)
        with pytest.raises(ValueError, match="Invalid bbox"):
            calculate_fixed_aspect_ratio_bbox((100, 100, 200, 100), (800, 600), (1, 1))

        # Invalid image dimensions
        with pytest.raises(ValueError, match="Invalid image dimensions"):
            calculate_fixed_aspect_ratio_bbox((100, 100, 200, 150), (0, 600), (1, 1))

        # Invalid aspect ratio
        with pytest.raises(ValueError, match="Invalid aspect ratio"):
            calculate_fixed_aspect_ratio_bbox((100, 100, 200, 150), (800, 600), (0, 1))

    def test_bbox_within_image_bounds(self):
        """Test bbox containment with new algorithm that may extend beyond image bounds."""
        test_cases = [
            # (original_bbox, image_dims, aspect_ratio)
            ((10, 10, 50, 30), (100, 100), (1, 1)),
            ((90, 90, 99, 99), (100, 100), (4, 3)),
            ((0, 0, 10, 10), (100, 100), (16, 9)),
            ((50, 50, 100, 80), (200, 150), (3, 2)),
        ]

        for original_bbox, image_dims, aspect_ratio in test_cases:
            result = calculate_fixed_aspect_ratio_bbox(original_bbox, image_dims, aspect_ratio)
            x1, y1, x2, y2 = result

            # Check that bbox has positive dimensions
            assert x2 > x1, f"Invalid bbox: x2 {x2} should be > x1 {x1}"
            assert y2 > y1, f"Invalid bbox: y2 {y2} should be > y1 {y1}"

            # Check containment of original bbox
            orig_x1, orig_y1, orig_x2, orig_y2 = original_bbox
            assert x1 <= orig_x1 and x2 >= orig_x2, f"X range {x1}-{x2} should contain original {orig_x1}-{orig_x2}"
            assert y1 <= orig_y1 and y2 >= orig_y2, f"Y range {y1}-{y2} should contain original {orig_y1}-{orig_y2}"

    def test_aspect_ratio_preservation(self):
        """Test that aspect ratios are preserved within reasonable tolerance."""
        test_ratios = [(1, 1), (4, 3), (16, 9), (3, 2), (21, 9), (9, 16)]

        for aspect_w, aspect_h in test_ratios:
            result = calculate_fixed_aspect_ratio_bbox((100, 100, 200, 150), (800, 600), (aspect_w, aspect_h))
            x1, y1, x2, y2 = result

            width = x2 - x1
            height = y2 - y1
            actual_ratio = width / height
            expected_ratio = aspect_w / aspect_h

            # Allow tolerance for integer rounding effects
            tolerance = 0.1
            assert abs(actual_ratio - expected_ratio) <= tolerance, \
                f"Aspect ratio {actual_ratio:.3f} not close enough to {expected_ratio:.3f} for {aspect_w}:{aspect_h}"

    def test_minimum_bbox_size(self):
        """Test that output bbox has minimum size of 1x1."""
        # Very small original bbox
        result = calculate_fixed_aspect_ratio_bbox((100, 100, 101, 101), (800, 600), (1, 1))
        x1, y1, x2, y2 = result

        width = x2 - x1
        height = y2 - y1
        assert width >= 1, f"Width too small: {width}"
        assert height >= 1, f"Height too small: {height}"

    def test_containment_over_area_preservation(self):
        """Test that containment is prioritized over area preservation."""
        original_bbox = (100, 100, 200, 150)  # 100x50 = 5000 area
        result = calculate_fixed_aspect_ratio_bbox(original_bbox, (800, 600), (1, 1))
        x1, y1, x2, y2 = result

        # Must contain the original bbox
        width = x2 - x1
        height = y2 - y1
        assert width >= 100, f"Width {width} should contain original width 100"
        assert height >= 50, f"Height {height} should contain original height 50"

        # Should be square (1:1 ratio)
        assert width == height, f"Should be square but got {width}x{height}"

        # With containment-first, area might be larger than original
        new_area = width * height
        original_area = 100 * 50  # 5000

        # Area should be at least as large as original (never smaller due to containment)
        assert new_area >= original_area, f"New area {new_area} should be >= original area {original_area}"

        # For this case, we expect a 100x100 square (area = 10000) to contain the 100x50 rectangle
        assert new_area == 10000, f"Expected 100x100 square (area 10000) but got area {new_area}"

    def test_docstring_examples(self):
        """Test examples from function docstring."""
        # Note: The examples in docstring are approximate, test with tolerance
        result1 = calculate_fixed_aspect_ratio_bbox((100, 100, 200, 150), (800, 600), (1, 1))
        x1, y1, x2, y2 = result1

        # Should be roughly square and centered
        width = x2 - x1
        height = y2 - y1
        assert width == height, "Should be square for 1:1 ratio"

        # Center should be approximately (150, 125)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        assert abs(center_x - 150) <= 5, f"Center X not close to 150: {center_x}"
        assert abs(center_y - 125) <= 5, f"Center Y not close to 125: {center_y}"

    def test_expand_to_contain_tall_source(self):
        """Test expanding tall source to 1:1 ratio."""
        # Source: 378×1115 (tall)
        original_bbox = (100, 100, 478, 1215)
        image_dims = (2000, 2000)  # Large image, no constraints
        aspect_ratio = (1, 1)

        result = calculate_fixed_aspect_ratio_bbox(original_bbox, image_dims, aspect_ratio)

        # Should expand to 1115×1115 to contain entire tall source
        expected_size = 1115  # max(378, 1115)
        x1, y1, x2, y2 = result
        width = x2 - x1
        height = y2 - y1

        assert width == expected_size
        assert height == expected_size
        assert abs(width / height - 1.0) < 0.01  # Perfect 1:1 ratio

        # Source should be fully contained
        orig_x1, orig_y1, orig_x2, orig_y2 = original_bbox
        assert x1 <= orig_x1
        assert x2 >= orig_x2
        assert y1 <= orig_y1
        assert y2 >= orig_y2

    def test_expand_to_contain_wide_source(self):
        """Test expanding wide source to 1:1 ratio."""
        # Source: 600×300 (wide)
        original_bbox = (100, 100, 700, 400)
        image_dims = (2000, 2000)  # Large image, no constraints
        aspect_ratio = (1, 1)

        result = calculate_fixed_aspect_ratio_bbox(original_bbox, image_dims, aspect_ratio)

        # Should expand to 600×600 to contain entire wide source
        expected_size = 600  # max(600, 300)
        x1, y1, x2, y2 = result
        width = x2 - x1
        height = y2 - y1

        assert width == expected_size
        assert height == expected_size
        assert abs(width / height - 1.0) < 0.01  # Perfect 1:1 ratio

        # Source should be fully contained
        orig_x1, orig_y1, orig_x2, orig_y2 = original_bbox
        assert x1 <= orig_x1
        assert x2 >= orig_x2
        assert y1 <= orig_y1
        assert y2 >= orig_y2

    def test_square_source_no_expansion(self):
        """Test that square source doesn't need expansion."""
        # Source: 400×400 (already square)
        original_bbox = (100, 100, 500, 500)
        image_dims = (2000, 2000)
        aspect_ratio = (1, 1)

        result = calculate_fixed_aspect_ratio_bbox(original_bbox, image_dims, aspect_ratio)

        # Should stay 400×400
        x1, y1, x2, y2 = result
        width = x2 - x1
        height = y2 - y1

        assert width == 400
        assert height == 400
        assert abs(width / height - 1.0) < 0.01  # Perfect 1:1 ratio

    def test_image_boundary_constraints(self):
        """Test behavior when ideal size exceeds image bounds with new algorithm."""
        # Source: 378×1115, but image is only 676×1280
        original_bbox = (32, 164, 410, 1279)
        image_dims = (676, 1280)
        aspect_ratio = (1, 1)

        result = calculate_fixed_aspect_ratio_bbox(original_bbox, image_dims, aspect_ratio)

        # With new algorithm, we get exact 1:1 containing the source
        x1, y1, x2, y2 = result
        width = x2 - x1
        height = y2 - y1

        # Source dimensions: 378×1115, so we need 1115×1115 to contain it
        source_width = 410 - 32  # 378
        source_height = 1279 - 164  # 1115
        expected_size = max(source_width, source_height)  # 1115

        assert width == expected_size, f"Expected width {expected_size} but got {width}"
        assert height == expected_size, f"Expected height {expected_size} but got {height}"
        assert abs(width / height - 1.0) < 0.01, "Should have perfect 1:1 ratio"

    def test_16_9_aspect_ratio(self):
        """Test with 16:9 aspect ratio."""
        # Source: 400×400 (square), want 16:9
        original_bbox = (100, 100, 500, 500)
        image_dims = (2000, 2000)
        aspect_ratio = (16, 9)

        result = calculate_fixed_aspect_ratio_bbox(original_bbox, image_dims, aspect_ratio)

        x1, y1, x2, y2 = result
        width = x2 - x1
        height = y2 - y1
        ratio = width / height
        target_ratio = 16 / 9

        # Should have exact 16:9 ratio
        assert abs(ratio - target_ratio) < 0.01

        # Should contain source (both dimensions >= 400)
        assert width >= 400
        assert height >= 400

    def test_never_shrinks_source(self):
        """Test that result never has dimensions smaller than source."""
        test_cases = [
            ((0, 0, 300, 600), (1, 1)),  # Tall source, square ratio
            ((0, 0, 600, 300), (1, 1)),  # Wide source, square ratio
            ((0, 0, 400, 400), (16, 9)), # Square source, wide ratio
            ((0, 0, 400, 400), (9, 16)), # Square source, tall ratio
        ]

        for original_bbox, aspect_ratio in test_cases:
            orig_x1, orig_y1, orig_x2, orig_y2 = original_bbox
            source_width = orig_x2 - orig_x1
            source_height = orig_y2 - orig_y1

            result = calculate_fixed_aspect_ratio_bbox(
                original_bbox, (2000, 2000), aspect_ratio
            )

            x1, y1, x2, y2 = result
            final_width = x2 - x1
            final_height = y2 - y1

            # Final dimensions should never be smaller than source
            assert final_width >= source_width, f"Width {final_width} < source {source_width}"
            assert final_height >= source_height, f"Height {final_height} < source {source_height}"


class TestValidateBbox:
    """Test bbox validation."""

    def test_valid_bbox(self):
        """Test validation of valid bboxes."""
        assert validate_bbox((10, 10, 50, 30), (100, 100)) is True
        assert validate_bbox((0, 0, 100, 100), (100, 100)) is True

    def test_invalid_bbox(self):
        """Test validation of invalid bboxes."""
        assert validate_bbox((50, 30, 10, 10), (100, 100)) is False  # x2 < x1, y2 < y1
        assert validate_bbox((10, 10, 110, 30), (100, 100)) is False  # x2 > image_width
        assert validate_bbox((-5, 10, 50, 30), (100, 100)) is False   # x1 < 0


class TestNormalizeBbox:
    """Test bbox normalization."""

    def test_clamp_to_bounds(self):
        """Test clamping coordinates to image bounds."""
        result = normalize_bbox((-10, -5, 50, 30), (100, 100))
        assert result == (0, 0, 50, 30)

    def test_fix_coordinate_ordering(self):
        """Test fixing swapped coordinates."""
        result = normalize_bbox((50, 30, 10, 10), (100, 100))
        assert result == (10, 10, 50, 30)

    def test_ensure_minimum_size(self):
        """Test ensuring minimum 1x1 size."""
        result = normalize_bbox((50, 50, 50, 50), (100, 100))
        assert result[2] > result[0]  # x2 > x1
        assert result[3] > result[1]  # y2 > y1
