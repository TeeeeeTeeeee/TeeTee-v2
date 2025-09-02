export const ZERO_ADDRESS = '0x0000000000000000000000000000000000000000';

export const toBigIntSafe = (n: number | string | bigint | undefined | null): bigint => {
  try {
    if (typeof n === 'bigint') return n;
    if (n === undefined || n === null || n === '') return 0n;
    return BigInt(n as any);
  } catch {
    return 0n;
  }
};
