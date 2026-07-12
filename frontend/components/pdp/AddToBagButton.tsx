"use client"

/**
 * AddToBagButton — client component for the mock PDP demo.
 *
 * The PDP page is a Server Component (it needs generateMetadata).
 * This tiny client island handles the onClick interaction so the server
 * component doesn't need to become a client component for one button.
 */

interface AddToBagButtonProps {
  /** Brand accent colour as a hex string, e.g. "#1a1a2e" */
  accentHex: string
}

export function AddToBagButton({ accentHex }: AddToBagButtonProps) {
  return (
    <button
      className="w-full py-3 rounded-xl text-white font-semibold text-base transition-opacity hover:opacity-90 active:opacity-75"
      style={{ backgroundColor: accentHex }}
      onClick={() => alert("This is a demo page — the button is non-functional.")}
    >
      Add to Bag
    </button>
  )
}
