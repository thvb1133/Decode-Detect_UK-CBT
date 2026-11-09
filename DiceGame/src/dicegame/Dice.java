package dicegame;

import java.util.Random;

// Two six-sided dice that share one random sequence.
// Sharing the sequence keeps a test seed repeatable.
public class Dice {

    private final Random random;

    Dice(Random random) {
        this.random = random;
    }

    // Each face is 1 to 6. The two rolls are independent.
    DiceThrow roll() {
        int first = random.nextInt(6) + 1;
        int second = random.nextInt(6) + 1;
        return new DiceThrow(first, second);
    }
}
