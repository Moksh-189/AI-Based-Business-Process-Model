
import { test, expect } from '@playwright/test';

test('Integration: Phase 4.2 Topology Dashboard', async ({ page }) => {
    // Navigate to Topology page
    await page.goto('/');
    await page.click('a[href="/topology"]');
    await page.waitForURL('**/topology');
    console.log('Navigated to /topology');

    // Wait for any content to load
    await page.waitForLoadState('networkidle');
    console.log('Network idle');

    // 1. Verify Process Topology Canvas Header
    await expect(page.getByText('Live Process Topology (SAP P2P)')).toBeVisible();

    // Verify nodes check (looking for specific labels)
    // We wait for the node to appear which confirms React Flow has rendered
    await expect(page.getByText('Start: Create Purchase Req')).toBeVisible();
    await expect(page.getByText('Clear Invoice (BOTTLENECK: 43.7d)')).toBeVisible();

    // 2. Verify Workforce Allocation Panel
    await expect(page.getByText('Workforce Allocation')).toBeVisible();
    // Check if draggables exist
    await expect(page.getByText('Alice Chen')).toBeVisible();

    // 3. Verify Telemetry Panel
    await expect(page.getByText('Live Telemetry')).toBeVisible();
    await expect(page.getByText('ROI +24% Proj.')).toBeVisible();

    // 4. Verify Recharts Graph
    const chart = page.locator('.recharts-responsive-container');
    await expect(chart).toBeVisible();
});
