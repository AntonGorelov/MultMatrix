#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <omp.h>

using namespace std;

double *A, *B, *C; // ������ ������ � ���� ���������� ��������
int N = 1440, M, K, S; // N - ����������� �������, M - ����������� �����, K = N / M - ���������� ������, S - ���������� ��������� � A � B

// ��������� ����� �� 0 �� 1 ���� float
float float_rand()
{
	return (float)rand() / (RAND_MAX);
}

// ��������� ����� �� 0 �� 1 ���� double
double double_rand()
{
	return (double)rand() / (RAND_MAX);
}

// ���������� ������� �
void matrixABlockPlacement()
{
	int cur = 0; // ����� �������� ��������
	for (int j1 = 0; j1 < N / M; j1++) // �������� ��� ������� ������� �������
		for (int i1 = 0; i1 <= j1; i1++) // �������� ��� ������� ������ �� ������� �� ������, ������� ������ �������� �������
			for (int i2 = 0; i2 < M; i2++) // �������� ��� ������ �����
				for (int j2 = 0; j2 < M; j2++) // �������� ��� ������� �����
					if (i1 * M + i2 > j1 * M + j2) // ���� ������� ����� ���� ������� ���������, �� �� ����� 0
						A[cur++] = 0;
					else
						A[cur++] = double_rand(); // ���� ���� ��� �� ���, �� ��������� ��������� ������
}

// ���������� ������� � ��������� ��������
void matrixAScalarPlacement()
{
	int cur = 0;
	for (int i1 = 0; i1 < N / M; i1++)
		for (int j1 = 0; j1 < M; j1++)
		{
			for (int i2 = 0; i2 < j1; i2++)
				A[cur++] = 0;
			for (int i2 = j1; i2 < N - i1 * M; i2++)
				A[cur++] = double_rand();
		}
}

// ���������� ������� �
void matrixBBlockPlacement()
{
	int cur = 0; // ����� �������� ��������
	for (int j1 = 0; j1 < N / M; j1++) // �������� ��� ������� ������� �������
		for (int i1 = j1; i1 < N / M; i1++) // �������� ��� ������� ������ �� ������, ������� ������ �������� �������, �� ���������
			for (int i2 = 0; i2 < M; i2++) // �������� ��� ������ �����
				for (int j2 = 0; j2 < M; j2++) // �������� ��� ������� �����
					if (i1 * M + i2 < j1 * M + j2) // ���� ������� ����� ���� ������� ���������, �� �� ����� 0
						B[cur++] = 0;
					else
						B[cur++] = double_rand(); // ���� ���� ��� �� ���, �� ��������� ��������� ������
}

// ���������� ������� B ��������� ��������
void loadBScalar()
{
	int cur = 0;
	for (int i1 = 0; i1 < N / M; i1++)
		for (int j1 = 0; j1 < M; j1++)
		{
			for (int i2 = 0; i2 < i1 * M + j1 + 1; i2++)
				B[cur++] = double_rand();
			for (int i2 = 0; i2 < M - j1 - 1; i2++)
				B[cur++] = 0;
		}
}


// ���������� �������-���������� ������
void matrixC()
{
	for (int i = 0; i < N * N; i++)
		C[i] = 0;
}

// ������������ ������ � ������� ����������� ������
void multMatrixBlockPlacement()
{
	for (int i1 = 0; i1 < N / M; i1++)
		for (int j1 = 0; j1 < N / M; j1++)
			for (int k1 = max(i1, j1); k1 < N / M; k1++) // ���� k1 < i1 ��� k1 < j1, �� ��������� ���������� �� ������� ����, ��� �� ������ �����.
				for (int i2 = 0; i2 < M; i2++)
					for (int j2 = 0; j2 < M; j2++)
						for (int k2 = 0; k2 < M; k2++)
							C[i1 * M * N + j1 * M * M + i2 * M + j2] +=
							A[(i1 + ((1 + k1) * k1 / 2)) * M * M + i2 * M + k2] 
							* B[((K - ((N / M - j1) * (N / M - j1 + 1) / 2)) + (k1 - j1)) * M * M + k2 * M + j2];
}

/* ����������������� ���������, ��� ������� ���� ������ ����� ������� A ���������� � ������ �����
�� ������ ������ ������� B*/ 
void multMatrixParallelInBlockSAMB()
{
   #pragma omp parallel num_threads(4)
   {
	   for (int i1 = 0; i1 < N / M; i1++)
		   for (int j1 = 0; j1 < N / M; j1++)
			   for (int k1 = max(i1, j1); k1 < N / M; k1++) // ���� k1 < i1 ��� k1 < j1, �� ��������� ���������� �� ������� ����, ��� �� ������ �����.
			   {
				   int thr = omp_get_thread_num(); // ����� �������� ������
				   for (int i2 = thr * M / 4; i2 < (thr + 1) * M / 4; i2++)
					   for (int j2 = 0; j2 < M; j2++)
						   for (int k2 = 0; k2 < M; k2++)
							   C[i1 * M * N + j1 * M * M + i2 * M + j2] +=
							   A[(i1 + ((1 + k1) * k1 / 2)) * M * M + i2 * M + k2]
							   * B[((K - ((N / M - j1) * (N / M - j1 + 1) / 2)) + (k1 - j1)) * M * M + k2 * M + j2];
			   }
   }
}

/* ����������������� ���������, ��� ������� ������ ������ ����� ������� A ���������� � ������ �����
�� ������ ������ ����� ������� B*/
void multMatrixParallelInBlockMAMB()
{
   #pragma omp parallel num_threads(4)
   {
	   for (int step = 0; step < 4; step++)
		   for (int i1 = 0; i1 < N / M; i1++)
			   for (int j1 = 0; j1 < N / M; j1++)
				   for (int k1 = max(i1, j1); k1 < N / M; k1++) // ���� k1 < i1 ��� k1 < j1, �� ��������� ���������� �� ������� ����, ��� �� ������ �����.
				   {
					   int thr = omp_get_thread_num(); // ����� �������� ������
					   for (int i2 = thr * M / 4; i2 < (thr + 1) * M / 4; i2++)
						   for (int j2 = ((step + thr) % 4) * M / 4; j2 < (((step + thr) % 4) + 1) * M / 4; j2++)
							   for (int k2 = 0; k2 < M; k2++)
								   C[i1 * M * N + j1 * M * M + i2 * M + j2] +=
								   A[(i1 + ((1 + k1) * k1 / 2)) * M * M + i2 * M + k2]
								   * B[((K - ((N / M - j1) * (N / M - j1 + 1) / 2)) + (k1 - j1)) * M * M + k2 * M + j2];
				   }
   }
}

void multMatrixParallelBetweenBlockSAMB()
{
#pragma omp parallel num_threads(min(N / M, 4))
{
	int thr = omp_get_thread_num();
	for (int i1 = thr * (N / M) / min(N / M, 4); i1 < (thr + 1) * (N / M) / min(N / M, 4); i1++)
		for (int j1 = 0; j1 < N / M; j1++)
			for (int k1 = max(i1, j1); k1 < N / M; k1++) // ���� k1 < i1 ��� k1 < j1, �� ��������� ���������� �� ������� ����, ��� �� ������ �����.
				for (int i2 = 0; i2 < M; i2++)
					for (int j2 = 0; j2 < M; j2++)
						for (int k2 = 0; k2 < M; k2++)
							C[i1 * M * N + j1 * M * M + i2 * M + j2] +=
							A[(i1 + ((1 + k1) * k1 / 2)) * M * M + i2 * M + k2]
							* B[((K - ((N / M - j1) * (N / M - j1 + 1) / 2)) + (k1 - j1)) * M * M + k2 * M + j2];
}
}

void multMatrixParallelBetweenBlockMAMB()
{
	for (int step = 0; step < min(N / M, 4); step++)
#pragma omp parallel num_threads(min(N / M, 4))
	{
		int thr = omp_get_thread_num();
		for (int i1 = thr * (N / M) / min(N / M, 4); i1 < (thr + 1) * (N / M) / min(N / M, 4); i1++)
			for (int j1 = ((step + thr) % 4) * (N / M) / min(N / M, 4); j1 < (((step + thr) % 4) + 1) * (N / M) / min(N / M, 4); j1++)
				for (int k1 = max(i1, j1); k1 < N / M; k1++) // ���� k1 < i1 ��� k1 < j1, �� ��������� ���������� �� ������� ����, ��� �� ������ �����.
					for (int i2 = 0; i2 < M; i2++)
						for (int j2 = 0; j2 < M; j2++)
							for (int k2 = 0; k2 < M; k2++)
								C[i1 * M * N + j1 * M * M + i2 * M + j2] +=
								A[(i1 + ((1 + k1) * k1 / 2)) * M * M + i2 * M + k2]
								* B[((K - ((N / M - j1) * (N / M - j1 + 1) / 2)) + (k1 - j1)) * M * M + k2 * M + j2];
	}
}

// ������������ ������, ���������� ��������
void multMatrixScalarPlacement()
{
	for (int i1 = 0; i1 < N / M; i1++)
		for (int j1 = 0; j1 < N / M; j1++)
			for (int k1 = max(i1, j1); k1 < N / M; k1++)
				for (int i2 = 0; i2 < M; i2++)
					for (int j2 = 0; j2 < M; j2++)
						for (int k2 = 0; k2 < M; k2++)
							C[i1 * M * N + i2 * N + j1 * M + j2] +=
							A[(M * M * (N / M * (N / M + 1) - (N / M - i1) * (N / M - i1 + 1)) / 2) + (k1 - i1) * M + i2 * (N - (i1 * M)) + k2]
							* B[(k1 * (k1 + 1) * M * M / 2) + k2 * M * (k1 + 1) + j1 * M + j2];
}

// ����� ��������
void print()
{
	for (int i = 0; i < S; i++)
		cout << A[i] << " ";
	cout << endl;
	for (int i = 0; i < S; i++)
		cout << B[i] << " ";
	cout << endl;
	for (int i = 0; i < N * N; i++)
		cout << C[i] << " ";
	cout << endl;
}

int main()
{
	srand(time(0));
	int sizes[] = { 1, 6, 10, 15, 20, 24, 30, 36, 40, 60, 72, 80, 96, 120, 144, 160, 180, 240, 360, 480, 720 };
	for (int i = 0; i < 21; i++)
	{
		M = sizes[i];
		K = (1 + (N / M)) * (N / M) / 2;
		S = K * M * M;
		A = new double[S];
		B = new double[S];
		C = new double[N * N];

		clock_t start, end;
		start = clock();
		matrixABlockPlacement();
		matrixBBlockPlacement();
		matrixC();
		multMatrixParallelBetweenBlockMAMB();
		end = clock();
		cout << "Size block: " << M << " Time: " << ((double)end - start) / ((double)CLOCKS_PER_SEC) << endl;

		delete[] A;
		delete[] B;
		delete[] C;
	}
}