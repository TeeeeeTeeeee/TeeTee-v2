import { useReadContract } from 'wagmi';
import { CONTRACT_ADDRESS } from '@/utils/address';
import ABI from '@/utils/abi.json';

export function useGetTotalUnclaimedEarnings(hostAddress?: `0x${string}`) {
  const { data, isLoading, error, refetch } = useReadContract({
    address: CONTRACT_ADDRESS as `0x${string}`,
    abi: ABI,
    functionName: 'getTotalUnclaimedEarnings',
    args: [hostAddress],
    query: {
      enabled: !!hostAddress,
    },
  });

  return {
    unclaimedEarnings: data as bigint | undefined,
    isLoading,
    error,
    refetch,
  };
}

