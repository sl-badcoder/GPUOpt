__kernel void globalSwap(
    const int i,
    const int j,
    __global int *arr,
    const int size)
{
    int x = get_global_id(0);
    if (x >= size) return;

    int mask = 1 << (i - j);
    int y    = x ^ mask;
    if (y > x && y < size) {
        int dir = (x & (1 << i)) != 0;
        int u   = arr[x], v = arr[y];
        if (dir) {
            if (u < v) {
                arr[x] = v;
                arr[y] = u;
            }
        } else {
            if (u > v) {
                arr[x] = v;
                arr[y] = u;
            }
        }
    }
}
