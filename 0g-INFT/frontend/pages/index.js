import Head from 'next/head'
import INFTDashboard from '../components/INFTDashboard'

export default function Home() {
  return (
    <>
      <Head>
        <title>0G INFT Dashboard - Intelligent NFTs on 0G Galileo Testnet</title>
        <meta name="description" content="Manage your Intelligent NFTs on 0G Galileo testnet. Mint, authorize, infer, and transfer INFTs with oracle-verified AI capabilities." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <INFTDashboard />
    </>
  )
}