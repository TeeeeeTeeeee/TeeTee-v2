import { useState, useMemo, type ReactNode } from 'react';
import type { NextPage } from 'next';
import { ConnectButton } from '@rainbow-me/rainbowkit';
import { useAccount } from 'wagmi';
import { formatEther } from 'viem';
import { CONTRACT_ADDRESS } from '@/utils/address';
import {
  toBigIntSafe,
  useCheckBundleAmount,
  useCheckBundlePrice,
  useCheckCreditPriceWei,
  useCheckOwner,
  useCheckUserCredits,
  useCheckHostedLLM,
  useCheckHostedLLMs,
  useCheckUserCreditsDirect,
  useBuyCredits,
  useUsePrompt,
  useRegisterLLM,
  useEditRegisteredLLM,
} from '@/lib/contracts/creditUse';

const Section = ({ title, children }: { title: string; children: ReactNode }) => (
  <section className="rounded-xl border border-gray-200 p-4 mb-4 bg-white shadow-sm">
    <h2 className="text-lg font-semibold mb-3">{title}</h2>
    <div className="space-y-3">{children}</div>
  </section>
);

const TestPage: NextPage = () => {
  const { address, isConnected } = useAccount();

  // Reads
  const { data: bundleAmount } = useCheckBundleAmount();
  const { data: bundlePrice } = useCheckBundlePrice();
  const { data: creditPriceWei } = useCheckCreditPriceWei();
  const { data: owner } = useCheckOwner();

  const { data: myCredits, refetch: refetchMyCredits } = useCheckUserCredits(address);

  // Write hooks
  const {
    buyCredits,
    txHash: buyTxHash,
    isWriting: isBuying,
    writeError: buyError,
    resetWrite: resetBuy,
    isConfirming: isBuyConfirming,
    isConfirmed: isBuyConfirmed,
  } = useBuyCredits();

  const { usePrompt, isWriting: isUsing, resetWrite: resetUsePrompt } = useUsePrompt();
  const { registerLLM, isWriting: isRegistering, resetWrite: resetRegister } = useRegisterLLM();
  const { editRegisteredLLM, isWriting: isEditing, resetWrite: resetEdit } = useEditRegisteredLLM();

  // Inputs
  const [bundlesToBuy, setBundlesToBuy] = useState<number>(1);
  const [usePromptId, setUsePromptId] = useState<string>('0');
  const [usePromptTokens, setUsePromptTokens] = useState<string>('1');
  const [lookupUser, setLookupUser] = useState<string>('');
  const [lookupId, setLookupId] = useState<string>('0');

  // Register LLM inputs
  const [regHost1, setRegHost1] = useState<string>('');
  const [regHost2, setRegHost2] = useState<string>('');
  const [regUrl1, setRegUrl1] = useState<string>('');
  const [regUrl2, setRegUrl2] = useState<string>('');
  const [regModel, setRegModel] = useState<string>('');

  // Edit LLM inputs
  const [editId, setEditId] = useState<string>('0');
  const [editHost1, setEditHost1] = useState<string>('');
  const [editHost2, setEditHost2] = useState<string>('');
  const [editUrl1, setEditUrl1] = useState<string>('');
  const [editUrl2, setEditUrl2] = useState<string>('');
  const [editModel, setEditModel] = useState<string>('');

  const totalCostWei = useMemo(() => {
    if (!bundlePrice) return 0n;
    const bundles = toBigIntSafe(bundlesToBuy || 0);
    return (bundlePrice as bigint) * bundles;
  }, [bundlePrice, bundlesToBuy]);

  const handleBuyCredits = async () => {
    if (!isConnected) return;
    resetBuy();
    try {
      await buyCredits(totalCostWei);
    } catch (e) {
      // handled via buyError
    }
  };

  const handleUsePrompt = async () => {
    if (!isConnected) return;
    resetUsePrompt();
    try {
      await usePrompt(toBigIntSafe(usePromptId), toBigIntSafe(usePromptTokens));
    } catch (e) {}
  };

  const handleRegisterLLM = async () => {
    if (!isConnected) return;
    resetRegister();
    try {
      await registerLLM(regHost1, regHost2, regUrl1, regUrl2, regModel);
    } catch (e) {}
  };

  const handleEditLLM = async () => {
    if (!isConnected) return;
    resetEdit();
    try {
      await editRegisteredLLM(toBigIntSafe(editId), editHost1, editHost2, editUrl1, editUrl2, editModel);
    } catch (e) {}
  };

  // Additional reads dependent on inputs
  const { data: lookupCredits, refetch: refetchLookupCredits } = useCheckUserCredits(lookupUser);
  const { data: hostedLLM } = useCheckHostedLLM(lookupId, lookupId !== '');
  const { data: hostedLLMDirect } = useCheckHostedLLMs(lookupId, lookupId !== '');
  const { data: userCreditsDirect, refetch: refetchUserCreditsDirect } = useCheckUserCreditsDirect(lookupUser);

  return (
    <main className="min-h-screen w-full bg-gray-50">
      <div className="max-w-3xl mx-auto p-6">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold">CreditUse Contract Tester</h1>
          <ConnectButton showBalance chainStatus="icon" accountStatus="address" />
        </div>

        <Section title="Contract Constants">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="p-3 rounded bg-gray-100">
              <div className="text-sm text-gray-500">BUNDLE_AMOUNT</div>
              <div className="font-mono">{bundleAmount?.toString?.() ?? '-'}</div>
            </div>
            <div className="p-3 rounded bg-gray-100">
              <div className="text-sm text-gray-500">BUNDLE_PRICE</div>
              <div className="font-mono">{bundlePrice ? `${formatEther(bundlePrice as bigint)} 0G` : '-'}</div>
            </div>
            <div className="p-3 rounded bg-gray-100">
              <div className="text-sm text-gray-500">CREDIT_PRICE_WEI</div>
              <div className="font-mono">{creditPriceWei ? `${formatEther(creditPriceWei as bigint)} 0G` : '-'}</div>
            </div>
            <div className="p-3 rounded bg-gray-100">
              <div className="text-sm text-gray-500">Owner</div>
              <div className="font-mono break-all">{(owner as string) ?? '-'}</div>
            </div>
          </div>
        </Section>

        <Section title="Your Credits">
          <div className="flex items-center gap-3">
            <div className="p-3 rounded bg-gray-100 flex-1">
              <div className="text-sm text-gray-500">Connected</div>
              <div className="font-mono break-all">{address ?? 'Not connected'}</div>
            </div>
            <div className="p-3 rounded bg-gray-100 flex-1">
              <div className="text-sm text-gray-500">checkUserCredits(you)</div>
              <div className="font-mono">{myCredits?.toString?.() ?? '-'}</div>
            </div>
          </div>
          <button
            className="mt-2 inline-flex items-center px-3 py-1.5 rounded bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50"
            onClick={() => refetchMyCredits?.()}
            disabled={!address}
          >
            Refresh
          </button>
        </Section>

        <Section title="Buy Credits (payable)">
          <div className="flex flex-wrap items-end gap-3">
            <label className="flex flex-col">
              <span className="text-sm text-gray-600">Bundles</span>
              <input
                type="number"
                min={1}
                value={bundlesToBuy}
                onChange={(e) => setBundlesToBuy(parseInt(e.target.value || '1'))}
                className="border rounded px-3 py-2 w-32"
              />
            </label>
            <div className="text-sm text-gray-600">
              Total cost: <span className="font-mono">{bundlePrice ? `${formatEther(totalCostWei)} 0G` : '-'}</span>
            </div>
            <button
              className="inline-flex items-center px-3 py-2 rounded bg-emerald-600 text-white hover:bg-emerald-700 disabled:opacity-50"
              onClick={handleBuyCredits}
              disabled={!isConnected || !bundlePrice || bundlesToBuy < 1 || isBuying}
            >
              {isBuying ? 'Sending…' : 'Buy Credits'}
            </button>
          </div>
          {buyError && <div className="text-red-600 text-sm">{(buyError as any)?.shortMessage || (buyError as Error).message}</div>}
          {buyTxHash && (
            <div className="text-sm text-gray-700">
              Tx: <span className="font-mono break-all">{buyTxHash}</span> {isBuyConfirming && '(confirming...)'} {isBuyConfirmed && '✅ Confirmed'}
            </div>
          )}
          {isBuyConfirmed && (
            <button className="mt-2 text-sm underline" onClick={() => refetchMyCredits?.()}>Refresh credits</button>
          )}
        </Section>

        <Section title="usePrompt(llmId, tokensUsed)">
          <div className="flex flex-wrap items-end gap-3">
            <label className="flex flex-col">
              <span className="text-sm text-gray-600">LLM ID</span>
              <input
                type="number"
                min={0}
                value={usePromptId}
                onChange={(e) => setUsePromptId(e.target.value)}
                className="border rounded px-3 py-2 w-40"
              />
            </label>
            <label className="flex flex-col">
              <span className="text-sm text-gray-600">Tokens Used</span>
              <input
                type="number"
                min={1}
                value={usePromptTokens}
                onChange={(e) => setUsePromptTokens(e.target.value)}
                className="border rounded px-3 py-2 w-40"
              />
            </label>
            <button
              className="inline-flex items-center px-3 py-2 rounded bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50"
              onClick={handleUsePrompt}
              disabled={!isConnected || isUsing}
            >
              {isUsing ? 'Sending…' : 'Use Prompt'}
            </button>
          </div>
        </Section>

        <Section title="getHostedLLM(id) / hostedLLMs(id) getter">
          <div className="flex flex-wrap items-end gap-3">
            <label className="flex flex-col">
              <span className="text-sm text-gray-600">ID</span>
              <input
                type="number"
                min={0}
                value={lookupId}
                onChange={(e) => setLookupId(e.target.value)}
                className="border rounded px-3 py-2 w-40"
              />
            </label>
          </div>
          {Boolean(hostedLLM) && (
            <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="p-3 rounded bg-gray-100"><span className="text-sm text-gray-600">host1</span><div className="font-mono break-all">{(hostedLLM as any).host1}</div></div>
              <div className="p-3 rounded bg-gray-100"><span className="text-sm text-gray-600">host2</span><div className="font-mono break-all">{(hostedLLM as any).host2}</div></div>
              <div className="p-3 rounded bg-gray-100"><span className="text-sm text-gray-600">shardUrl1</span><div className="font-mono break-all">{(hostedLLM as any).shardUrl1}</div></div>
              <div className="p-3 rounded bg-gray-100"><span className="text-sm text-gray-600">shardUrl2</span><div className="font-mono break-all">{(hostedLLM as any).shardUrl2}</div></div>
              <div className="p-3 rounded bg-gray-100"><span className="text-sm text-gray-600">modelName</span><div className="font-mono break-all">{(hostedLLM as any).modelName}</div></div>
              <div className="p-3 rounded bg-gray-100"><span className="text-sm text-gray-600">poolBalance</span><div className="font-mono break-all">{(hostedLLM as any).poolBalance?.toString?.()}</div></div>
            </div>
          )}

          {Boolean(hostedLLMDirect) && (
            <div className="mt-3">
              <div className="text-sm text-gray-600">Direct public getter hostedLLMs(id):</div>
              <div className="mt-1 grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div className="p-3 rounded bg-gray-100"><span className="text-sm text-gray-600">host1</span><div className="font-mono break-all">{(hostedLLMDirect as any).host1}</div></div>
                <div className="p-3 rounded bg-gray-100"><span className="text-sm text-gray-600">host2</span><div className="font-mono break-all">{(hostedLLMDirect as any).host2}</div></div>
                <div className="p-3 rounded bg-gray-100"><span className="text-sm text-gray-600">shardUrl1</span><div className="font-mono break-all">{(hostedLLMDirect as any).shardUrl1}</div></div>
                <div className="p-3 rounded bg-gray-100"><span className="text-sm text-gray-600">shardUrl2</span><div className="font-mono break-all">{(hostedLLMDirect as any).shardUrl2}</div></div>
                <div className="p-3 rounded bg-gray-100"><span className="text-sm text-gray-600">modelName</span><div className="font-mono break-all">{(hostedLLMDirect as any).modelName}</div></div>
                <div className="p-3 rounded bg-gray-100"><span className="text-sm text-gray-600">poolBalance</span><div className="font-mono break-all">{(hostedLLMDirect as any).poolBalance?.toString?.()}</div></div>
              </div>
            </div>
          )}
        </Section>

        <Section title="registerLLM(host1, host2, shardUrl1, shardUrl2, modelName) — owner only">
          <div className="flex flex-col gap-3">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <label className="flex flex-col">
                <span className="text-sm text-gray-600">Host 1</span>
                <input
                  type="text"
                  placeholder="0x..."
                  value={regHost1}
                  onChange={(e) => setRegHost1(e.target.value)}
                  className="border rounded px-3 py-2"
                />
              </label>
              <label className="flex flex-col">
                <span className="text-sm text-gray-600">Host 2</span>
                <input
                  type="text"
                  placeholder="0x..."
                  value={regHost2}
                  onChange={(e) => setRegHost2(e.target.value)}
                  className="border rounded px-3 py-2"
                />
              </label>
              <label className="flex flex-col">
                <span className="text-sm text-gray-600">Shard URL 1</span>
                <input
                  type="text"
                  placeholder="https://..."
                  value={regUrl1}
                  onChange={(e) => setRegUrl1(e.target.value)}
                  className="border rounded px-3 py-2"
                />
              </label>
              <label className="flex flex-col">
                <span className="text-sm text-gray-600">Shard URL 2</span>
                <input
                  type="text"
                  placeholder="https://..."
                  value={regUrl2}
                  onChange={(e) => setRegUrl2(e.target.value)}
                  className="border rounded px-3 py-2"
                />
              </label>
              <label className="flex flex-col sm:col-span-2">
                <span className="text-sm text-gray-600">Model Name</span>
                <input
                  type="text"
                  placeholder="Model name"
                  value={regModel}
                  onChange={(e) => setRegModel(e.target.value)}
                  className="border rounded px-3 py-2"
                />
              </label>
            </div>
            <div>
              <button
                className="inline-flex items-center px-3 py-2 rounded bg-rose-600 text-white hover:bg-rose-700 disabled:opacity-50"
                onClick={handleRegisterLLM}
                disabled={!isConnected || isRegistering || !regHost1 || !regHost2 || !regUrl1 || !regUrl2 || !regModel}
              >
                {isRegistering ? 'Sending…' : 'Register LLM'}
              </button>
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-2">Note: Owner-only. Provide valid host addresses and non-empty URLs and model name.</p>
        </Section>

        <Section title="editRegistedLLM(id, host1, host2, shardUrl1, shardUrl2, modelName) — owner only">
          <div className="flex flex-col gap-3">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <label className="flex flex-col">
                <span className="text-sm text-gray-600">LLM ID</span>
                <input
                  type="number"
                  min={0}
                  value={editId}
                  onChange={(e) => setEditId(e.target.value)}
                  className="border rounded px-3 py-2"
                />
              </label>
              <label className="flex flex-col">
                <span className="text-sm text-gray-600">Host 1 (optional)</span>
                <input
                  type="text"
                  placeholder="0x... (leave empty to keep current)"
                  value={editHost1}
                  onChange={(e) => setEditHost1(e.target.value)}
                  className="border rounded px-3 py-2"
                />
              </label>
              <label className="flex flex-col">
                <span className="text-sm text-gray-600">Host 2 (optional)</span>
                <input
                  type="text"
                  placeholder="0x... (leave empty to keep current)"
                  value={editHost2}
                  onChange={(e) => setEditHost2(e.target.value)}
                  className="border rounded px-3 py-2"
                />
              </label>
              <label className="flex flex-col">
                <span className="text-sm text-gray-600">Shard URL 1 (optional)</span>
                <input
                  type="text"
                  placeholder="https://... (leave empty to keep current)"
                  value={editUrl1}
                  onChange={(e) => setEditUrl1(e.target.value)}
                  className="border rounded px-3 py-2"
                />
              </label>
              <label className="flex flex-col">
                <span className="text-sm text-gray-600">Shard URL 2 (optional)</span>
                <input
                  type="text"
                  placeholder="https://... (leave empty to keep current)"
                  value={editUrl2}
                  onChange={(e) => setEditUrl2(e.target.value)}
                  className="border rounded px-3 py-2"
                />
              </label>
              <label className="flex flex-col sm:col-span-2">
                <span className="text-sm text-gray-600">Model Name (optional)</span>
                <input
                  type="text"
                  placeholder="Model name (leave empty to keep current)"
                  value={editModel}
                  onChange={(e) => setEditModel(e.target.value)}
                  className="border rounded px-3 py-2"
                />
              </label>
            </div>
            <div>
              <button
                className="inline-flex items-center px-3 py-2 rounded bg-orange-600 text-white hover:bg-orange-700 disabled:opacity-50"
                onClick={handleEditLLM}
                disabled={!isConnected || isEditing || editId === ''}
              >
                {isEditing ? 'Sending…' : 'Edit LLM'}
              </button>
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-2">Note: Owner-only. Leave fields empty to keep current values. Only non-zero addresses and non-empty strings will update the corresponding fields.</p>
        </Section>

        <Section title="checkUserCredits(address)">
          <div className="flex flex-wrap items-end gap-3">
            <label className="flex flex-col flex-1 min-w-[280px]">
              <span className="text-sm text-gray-600">Address</span>
              <input
                type="text"
                placeholder="0x..."
                value={lookupUser}
                onChange={(e) => setLookupUser(e.target.value.trim())}
                className="border rounded px-3 py-2 w-full"
              />
            </label>
            <button
              className="inline-flex items-center px-3 py-2 rounded bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50"
              onClick={() => { refetchLookupCredits?.(); refetchUserCreditsDirect?.(); }}
              disabled={!lookupUser || lookupUser.length !== 42}
            >
              Lookup
            </button>
          </div>
          <div className="mt-2">
            <span className="text-sm text-gray-600">Credits:</span>{' '}
            <span className="font-mono">{lookupCredits?.toString?.() ?? '-'}</span>
          </div>
        </Section>

        <p className="text-xs text-gray-500">Chain must match the one configured in Providers. Contract: <span className="font-mono">{CONTRACT_ADDRESS}</span></p>
      </div>
    </main>
  );
};

export default TestPage;
