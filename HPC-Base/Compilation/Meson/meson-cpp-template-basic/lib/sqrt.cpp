unsigned int sqrt(unsigned int x)
{
	unsigned int L = 0;
	unsigned int M;
	unsigned int R = x + 1;

    while (L != R - 1)
    {
        M = (L + R) / 2;

		if (M * M <= x)
			L = M;
		else
			R = M;
	}

    return L;
}