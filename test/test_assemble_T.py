# test_my_module.py

import unittest
import torch
from src.models.bfgpc import _assemble_T # Make sure to import from the correct file name

class TestAssembleT(unittest.TestCase):

    def test_standard_case(self):
        """
        Tests a standard scenario with L, H, and prediction points,
        including co-located points.
        """
        # Scenario:
        # 3 low-fidelity points (X_L)
        # 2 high-fidelity points (X_H)
        # 1 prediction point (X_prime)
        # One X_H is co-located with an X_L.
        # The X_prime is co-located with a different X_L.
        N_L = 3
        N_H = 2
        N_prime = 1
        rho_val = 0.9

        # Locations:
        # X_L at locs [A, B, C]
        # X_H at locs [A, D]
        # X_prime at loc [B]
        # Unique f_L locations are [A, B, C, D]
        N_f_L_unique = 4
        inverse_indices_L = torch.tensor([
            0, 1, 2,  # X_L maps to f_L unique locs A, B, C
            0,       # First X_H maps to f_L unique loc A
            3,       # Second X_H maps to f_L unique loc D
            1,       # X_prime maps to f_L unique loc B
        ])

        # N_f_delta corresponds to unique locations in H and prime
        N_f_delta = N_H + N_prime # 2 + 1 = 3

        # Expected T matrix shape: (3+2+1) x (4+3) = 6x7
        expected_T = torch.zeros((6, 7))

        # Block 1: N_L rows (rows 0, 1, 2)
        expected_T[0, 0] = 1.0  # L1 -> f_L(A)
        expected_T[1, 1] = 1.0  # L2 -> f_L(B)
        expected_T[2, 2] = 1.0  # L3 -> f_L(C)

        # Block 2: N_H rows (rows 3, 4)
        # H1 -> rho*f_L(A) + delta_1
        expected_T[3, 0] = rho_val
        expected_T[3, N_f_L_unique + 0] = 1.0
        # H2 -> rho*f_L(D) + delta_2
        expected_T[4, 3] = rho_val
        expected_T[4, N_f_L_unique + 1] = 1.0

        # Block 3: N_prime rows (row 5)
        # P1 -> rho*f_L(B) + delta_3
        expected_T[5, 1] = rho_val
        expected_T[5, N_f_L_unique + N_H + 0] = 1.0

        T = _assemble_T(N_H, N_L, N_f_L_unique, N_f_delta, N_prime, inverse_indices_L, rho_val)

        self.assertTrue(torch.equal(T, expected_T))
        self.assertEqual(T.shape, (6, 7))


    def test_no_high_fidelity_points(self):
        """Tests the case where N_H = 0."""
        N_L = 2
        N_H = 0
        N_prime = 1
        rho_val = 0.8
        N_f_L_unique = 2
        # X_L at [A, B], X_prime at [A]
        inverse_indices_L = torch.tensor([0, 1, 0])
        N_f_delta = N_H + N_prime # 0 + 1 = 1

        # Expected shape: (2+0+1) x (2+1) = 3x3
        expected_T = torch.zeros((3, 3))
        # Block 1 (L)
        expected_T[0, 0] = 1.0
        expected_T[1, 1] = 1.0
        # Block 3 (prime)
        expected_T[2, 0] = rho_val
        expected_T[2, N_f_L_unique + N_H + 0] = 1.0 # col index = 2+0+0 = 2

        T = _assemble_T(N_H, N_L, N_f_L_unique, N_f_delta, N_prime, inverse_indices_L, rho_val)
        self.assertTrue(torch.equal(T, expected_T))


    def test_no_prediction_points(self):
        """Tests the case where N_prime = 0."""
        N_L = 1
        N_H = 2
        N_prime = 0
        rho_val = 0.7
        N_f_L_unique = 3
        # X_L at [A], X_H at [B, C]
        inverse_indices_L = torch.tensor([0, 1, 2])
        N_f_delta = N_H + N_prime # 2 + 0 = 2

        # Expected shape: (1+2+0) x (3+2) = 3x5
        expected_T = torch.zeros((3, 5))
        # Block 1 (L)
        expected_T[0, 0] = 1.0
        # Block 2 (H)
        expected_T[1, 1] = rho_val
        expected_T[1, N_f_L_unique + 0] = 1.0 # col index = 3
        expected_T[2, 2] = rho_val
        expected_T[2, N_f_L_unique + 1] = 1.0 # col index = 4

        T = _assemble_T(N_H, N_L, N_f_L_unique, N_f_delta, N_prime, inverse_indices_L, rho_val)
        self.assertTrue(torch.equal(T, expected_T))

    def test_only_low_fidelity(self):
        """Tests the case with only low-fidelity points."""
        N_L = 3
        N_H = 0
        N_prime = 0
        rho_val = 0.5 # rho_val is unused here
        N_f_L_unique = 3
        inverse_indices_L = torch.tensor([0, 1, 2])
        N_f_delta = N_H + N_prime # 0

        # Expected shape: (3+0+0) x (3+0) = 3x3
        expected_T = torch.eye(3)

        T = _assemble_T(N_H, N_L, N_f_L_unique, N_f_delta, N_prime, inverse_indices_L, rho_val)
        self.assertTrue(torch.equal(T, expected_T))

    def test_empty_inputs(self):
        """Tests the case where all Ns are zero."""
        N_L, N_H, N_prime = 0, 0, 0
        N_f_L_unique = 0
        N_f_delta = 0
        inverse_indices_L = torch.tensor([])
        rho_val = 0.9

        expected_T = torch.zeros((0, 0))
        T = _assemble_T(N_H, N_L, N_f_L_unique, N_f_delta, N_prime, inverse_indices_L, rho_val)

        self.assertTrue(torch.equal(T, expected_T))
        self.assertEqual(T.shape, (0, 0))

    def test_rho_is_zero(self):
        """Tests that rho_val=0 correctly zeros out entries."""
        N_L, N_H, N_prime = 1, 1, 1
        rho_val = 0.0
        N_f_L_unique = 2
        inverse_indices_L = torch.tensor([0, 1, 0])
        N_f_delta = N_H + N_prime # 2

        # Expected shape: (1+1+1) x (2+2) = 3x4
        expected_T = torch.zeros((3, 4))
        # Block 1 (L)
        expected_T[0, 0] = 1.0
        # Block 2 (H)
        expected_T[1, 1] = 0.0 # rho_val
        expected_T[1, N_f_L_unique + 0] = 1.0 # col 2
        # Block 3 (prime)
        expected_T[2, 0] = 0.0 # rho_val
        expected_T[2, N_f_L_unique + N_H + 0] = 1.0 # col 3

        T = _assemble_T(N_H, N_L, N_f_L_unique, N_f_delta, N_prime, inverse_indices_L, rho_val)
        self.assertTrue(torch.equal(T, expected_T))


    def test_assertion_error_duplicate_rows(self):
        """
        Tests that an AssertionError is raised when inputs would create
        duplicate rows, which the function is designed to prevent.
        """
        # Scenario: 2 low-fidelity points at the exact same location.
        N_L = 2
        N_H = 0
        N_prime = 0
        N_f_L_unique = 1 # Both points map to the same unique location
        inverse_indices_L = torch.tensor([0, 0])
        N_f_delta = 0
        rho_val = 0.9

        # This setup will create T = [[1.0], [1.0]].
        # torch.unique(T, dim=0) will have a length of 1,
        # while len(T) is 2. This should trigger the assertion.
        with self.assertRaises(AssertionError):
            _assemble_T(N_H, N_L, N_f_L_unique, N_f_delta, N_prime, inverse_indices_L, rho_val)


if __name__ == '__main__':
    unittest.main()