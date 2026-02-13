
import { test, expect } from '@playwright/test';

test('Verify Draggable Cards', async ({ page }) => {
    console.log("Starting DnD check...");
    await page.goto('http://localhost:5173/topology');

    // Wait for component text
    await expect(page.getByText('Workforce Allocation')).toBeVisible();
    console.log("Workforce component visible");

    // Check for cards
    const cards = page.getByTestId('draggable-card');
    await expect(cards.first()).toBeVisible();

    // Count them
    const count = await cards.count();
    console.log(`Found ${count} draggable cards`);

    expect(count).toBeGreaterThan(0);

    const alice = page.getByText('Alice Chen');
    await expect(alice).toBeVisible();

    // Perform Drag and Drop
    await alice.dragTo(page.locator('.min-h-\\[120px\\]').first());

    // Verify Toast appears
    await expect(page.getByText('Digital Twin simulation updated.')).toBeVisible({ timeout: 10000 });
    console.log("DnD Toast verified");
});
