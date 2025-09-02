import { useReadContract } from 'wagmi';
import ABI from '@/utils/abi.json';
import { CONTRACT_ADDRESS } from '@/utils/address';
import { ZERO_ADDRESS } from '../utils';

export const useCheckUserCreditsDirect = (user?: string, enabled: boolean = Boolean(user && user.length === 42)) =>
  useReadContract({
    abi: ABI as any,
    address: CONTRACT_ADDRESS as `0x${string}`,
    functionName: 'userCredits',
    args: [user || ZERO_ADDRESS],
    query: { enabled },
  });

export default useCheckUserCreditsDirect;
