'use client'

import React, { useMemo, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Label } from '../components/ui/label'
import { Input } from '../components/ui/input'
import { Button } from '../components/ui/button'
import { Separator } from '../components/ui/separator'

export default function TokenURIBuilder() {
  const [name, setName] = useState('INFT #1')
  const [description, setDescription] = useState('Intelligent NFT backed by encrypted data on 0G Storage.')
  const [image, setImage] = useState('https://')
  const [attributes, setAttributes] = useState([{ trait_type: 'Model', value: 'Llama3.2' }])
  const [encryptedUri, setEncryptedUri] = useState('0g://storage/...')
  const [storageRootHash, setStorageRootHash] = useState('0x...')
  const [metadataHash, setMetadataHash] = useState('0x...')

  const [nftStorageToken, setNftStorageToken] = useState('')
  const [uploading, setUploading] = useState(false)
  const [cid, setCid] = useState('')
  const [error, setError] = useState('')

  const metadata = useMemo(() => {
    return {
      name: name.trim(),
      description: description.trim(),
      image: image.trim(),
      attributes: attributes
        .filter((a) => a.trait_type && a.value)
        .map((a) => ({ trait_type: String(a.trait_type), value: String(a.value) })),
      encrypted_uri: encryptedUri.trim(),
      storage_root_hash: storageRootHash.trim(),
      metadata_hash: metadataHash.trim()
    }
  }, [name, description, image, attributes, encryptedUri, storageRootHash, metadataHash])

  const jsonString = useMemo(() => JSON.stringify(metadata, null, 2), [metadata])
  const tokenUri = cid ? `ipfs://${cid}` : ''
  const gatewayUrl = cid ? `https://ipfs.io/ipfs/${cid}` : ''

  const addAttribute = () => {
    setAttributes((prev) => [...prev, { trait_type: '', value: '' }])
  }

  const updateAttribute = (index, key, value) => {
    setAttributes((prev) => prev.map((a, i) => (i === index ? { ...a, [key]: value } : a)))
  }

  const removeAttribute = (index) => {
    setAttributes((prev) => prev.filter((_, i) => i !== index))
  }

  const copy = async (text) => {
    try {
      await navigator.clipboard.writeText(text)
      alert('Copied to clipboard')
    } catch {}
  }

  const downloadJson = () => {
    const blob = new Blob([jsonString], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'metadata.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  const uploadToIpfs = async () => {
    setError('')
    setCid('')
    if (!nftStorageToken) {
      setError('NFT.Storage API token is required to upload')
      return
    }
    if (!metadata.name || !metadata.description || !metadata.image) {
      setError('Please fill name, description, and image')
      return
    }
    try {
      setUploading(true)
      const blob = new Blob([jsonString], { type: 'application/json' })
      const res = await fetch('https://api.nft.storage/upload', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${nftStorageToken}`
        },
        body: blob
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || `Upload failed: ${res.status}`)
      }
      const data = await res.json()
      if (!data || !data.ok || !data.value || !data.value.cid) {
        throw new Error('Unexpected response from NFT.Storage')
      }
      setCid(data.value.cid)
    } catch (e) {
      setError(e.message || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-4xl mx-auto px-4 py-8">
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="text-base">Build Token URI (ERC‑721 Metadata JSON)</CardTitle>
            <CardDescription className="text-xs">Creates public metadata and embeds your 0G fields</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="name" className="text-xs">Name</Label>
                <Input id="name" value={name} onChange={(e) => setName(e.target.value)} placeholder="INFT #1" className="text-sm" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="image" className="text-xs">Image URL (https/ipfs)</Label>
                <Input id="image" value={image} onChange={(e) => setImage(e.target.value)} placeholder="ipfs://.../image.png" className="text-sm" />
              </div>
              <div className="md:col-span-2 space-y-2">
                <Label htmlFor="description" className="text-xs">Description</Label>
                <Input id="description" value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Description" className="text-sm" />
              </div>
            </div>

            <div className="space-y-2">
              <div className="text-xs font-medium">Attributes</div>
              <div className="space-y-2">
                {attributes.map((attr, idx) => (
                  <div key={idx} className="grid grid-cols-2 gap-2">
                    <Input
                      value={attr.trait_type}
                      onChange={(e) => updateAttribute(idx, 'trait_type', e.target.value)}
                      placeholder="trait_type"
                      className="text-sm"
                    />
                    <div className="flex gap-2">
                      <Input
                        value={attr.value}
                        onChange={(e) => updateAttribute(idx, 'value', e.target.value)}
                        placeholder="value"
                        className="text-sm"
                      />
                      <Button type="button" variant="ghost" size="sm" onClick={() => removeAttribute(idx)}>Remove</Button>
                    </div>
                  </div>
                ))}
                <Button type="button" size="sm" onClick={addAttribute}>Add attribute</Button>
              </div>
            </div>

            <Separator />

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label htmlFor="encryptedUri" className="text-xs">encrypted_uri (0G link)</Label>
                <Input id="encryptedUri" value={encryptedUri} onChange={(e) => setEncryptedUri(e.target.value)} placeholder="0g://storage/<rootHash>" className="text-sm" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="root" className="text-xs">storage_root_hash</Label>
                <Input id="root" value={storageRootHash} onChange={(e) => setStorageRootHash(e.target.value)} placeholder="0x..." className="text-sm" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="metaHash" className="text-xs">metadata_hash (keccak256 of encrypted payload)</Label>
                <Input id="metaHash" value={metadataHash} onChange={(e) => setMetadataHash(e.target.value)} placeholder="0x..." className="text-sm" />
              </div>
            </div>

            <div className="space-y-2">
              <div className="text-xs">JSON Preview</div>
              <pre className="text-xs bg-card border border-border rounded p-3 overflow-auto max-h-72 whitespace-pre-wrap">{jsonString}</pre>
              <div className="flex gap-2">
                <Button type="button" size="sm" onClick={() => copy(jsonString)}>Copy JSON</Button>
                <Button type="button" size="sm" variant="secondary" onClick={downloadJson}>Download JSON</Button>
              </div>
            </div>

            <Separator />

            <div className="space-y-2">
              <Label htmlFor="token" className="text-xs">NFT.Storage API Token (optional)</Label>
              <Input id="token" value={nftStorageToken} onChange={(e) => setNftStorageToken(e.target.value)} placeholder="Paste token to upload to IPFS" className="text-sm" />
              <div className="flex gap-2">
                <Button type="button" size="sm" onClick={uploadToIpfs} disabled={uploading}>{uploading ? 'Uploading...' : 'Upload to IPFS'}</Button>
                {cid && (
                  <>
                    <Button type="button" size="sm" variant="secondary" onClick={() => copy(tokenUri)}>Copy tokenURI</Button>
                    <Button type="button" size="sm" variant="ghost" onClick={() => window.open(gatewayUrl, '_blank')}>Open on gateway</Button>
                  </>
                )}
              </div>
              {tokenUri && (
                <div className="text-xs text-muted-foreground">
                  tokenURI: <code className="font-mono">{tokenUri}</code>
                </div>
              )}
              {error && (
                <div className="text-xs text-destructive">{error}</div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}


