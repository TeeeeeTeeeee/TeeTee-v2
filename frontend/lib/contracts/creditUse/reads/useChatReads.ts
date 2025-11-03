import { useReadContract } from 'wagmi';
import { CONTRACT_ADDRESS } from '@/utils/address';
import ABI from '@/utils/abi.json';

export function useUserSessionCount(address?: `0x${string}`) {
  const { data, isLoading, error, refetch } = useReadContract({
    address: CONTRACT_ADDRESS as `0x${string}`,
    abi: ABI,
    functionName: 'userSessionCount',
    args: address ? [address] : undefined,
    query: {
      enabled: !!address,
    },
  });

  return {
    sessionCount: data as bigint | undefined,
    isLoading,
    error,
    refetch,
  };
}

export function useGetSessionMessages(sessionId: bigint, address?: `0x${string}`) {
  const { data, isLoading, error, refetch } = useReadContract({
    address: CONTRACT_ADDRESS as `0x${string}`,
    abi: ABI,
    functionName: 'getSessionMessages',
    args: [sessionId],
    account: address,
    query: {
      enabled: !!address && sessionId !== undefined,
    },
  });

  return {
    messages: data as any[] | undefined,
    isLoading,
    error,
    refetch,
  };
}

export function useGetSessionMessageCount(sessionId: bigint, address?: `0x${string}`) {
  const { data, isLoading, error, refetch } = useReadContract({
    address: CONTRACT_ADDRESS as `0x${string}`,
    abi: ABI,
    functionName: 'getSessionMessageCount',
    args: [sessionId],
    account: address,
    query: {
      enabled: !!address && sessionId !== undefined,
    },
  });

  return {
    messageCount: data as bigint | undefined,
    isLoading,
    error,
    refetch,
  };
}

