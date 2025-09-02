import { useReadContract } from 'wagmi';
import ABI from '@/utils/abi.json';
import { CONTRACT_ADDRESS } from '@/utils/address';
import { toBigIntSafe } from '../utils';

export const useCheckHostedLLMs = (id: number | string | bigint, enabled: boolean = id !== undefined && id !== null && `${id}` !== '') =>
  useReadContract({
    abi: ABI as any,
    address: CONTRACT_ADDRESS as `0x${string}`,
    functionName: 'hostedLLMs',
    args: [toBigIntSafe(id)],
    query: { enabled },
  });

export default useCheckHostedLLMs;
