import { useReadContract } from 'wagmi';
import { CONTRACT_ADDRESS } from '@/utils/address';
import ABI from '@/utils/abi.json';

export function useCheckWithdrawable(llmId: bigint, hostAddress?: `0x${string}`) {
  const { data, isLoading, error, refetch } = useReadContract({
    address: CONTRACT_ADDRESS as `0x${string}`,
    abi: ABI,
    functionName: 'checkWithdrawable',
    args: [llmId, hostAddress],
    query: {
      enabled: !!hostAddress && llmId !== undefined,
    },
  });

  return {
    withdrawableAmount: data as bigint | undefined,
    isLoading,
    error,
    refetch,
  };
}

