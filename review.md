# matrix multiply

## Coppersmith-Winograd

$$
    C = A * B
$$


分块为:
$$ 
A=
\left[
 \begin{matrix}
   A11 & A12 \\
   A21 & A22
  \end{matrix}
\right]
$$

$$ B=
\left[
 \begin{matrix}
   B11 & B12 \\
   B21 & B22
  \end{matrix}
\right]
$$

$$ C=
\left[
 \begin{matrix}
   C11 & C12 \\
   C21 & C22
  \end{matrix}
\right]
$$

$$
S1 = A21 + A22\\
S2 = S1 - A11\\
S3 = A11 - A21\\
S4 = A12 - S2\\

T1 = B12 - B11 \\
T2 = B22 - T1 \\
T3 = B22 - B12 \\
T4 = T2 - B21 \\

M1 = A11 * B11\\
M2 = A12 * B21\\
M3 = S4 * B22\\
M4 = A22 * T4\\
M5 = S1 * T1 \\
M6 = S2 * T2 \\
M7 = S3 * T3 \\

U1 = M1 + M2 \\
U2 = M1 + M6 \\
U3 = U2 + M7 \\
U4 = U2 + M5 \\
U5 = U4 + M3 \\
U6 = U3 - U4 \\
U7 = U3 + M5 \\

C11 = U1 \\
C12 = U5 \\
C21 = U6 \\
C22 = U7 \\

$$