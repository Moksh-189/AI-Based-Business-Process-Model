import { test, expect } from '@playwright/test';

test('UI Smoke Test: HUD & Navigation', async ({ page }) => {
    // 1. Visit the Home Page
    await page.goto('/');

    // 2. Verify Title and Hero content
    await expect(page).toHaveTitle(/AI.BPI - Digital Twin/);
    await page.waitForSelector('h1');
    await expect(page.locator('h1')).toContainText('Digital Twin');
    await expect(page.locator('h1')).toContainText('Simulation');

    // 3. Verify Sidebar Navigation exists
    const sidebar = page.locator('nav');
    await expect(sidebar).toBeVisible();

    // 4. Verify HUD "System Online" indicator
    await expect(page.getByText('System Online')).toBeVisible();

    // 5. Verify Floating Chatbot is present (and closed initially)
    const chatButton = page.locator('button > svg.lucide-message-square');
    await expect(chatButton).toBeVisible();

    // 6. Navigate to Topology page
    await page.click('a[href="/topology"]');
    await expect(page.locator('text=Process Topology (Phase 4.2)')).toBeVisible();
});
