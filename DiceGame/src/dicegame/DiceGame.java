package dicegame;

import java.util.Random;

// Beejalben Amitkumar Patel
// M01035595
// CST2110 Task 1

// Starts the console game. An optional first argument is a dice seed,
// used only to repeat the same throws while testing. Normal play omits it.
public class DiceGame {

    public static void main(String[] args) {
        Random random = new Random();
        if (args.length > 0) {
            try {
                random = new Random(Long.parseLong(args[0]));
            } catch (NumberFormatException ex) {
                System.out.println("The dice seed was not a whole number, so the dice will not be fixed.");
            }
        }

        Game game = new Game(System.in, System.out, random);
        game.play();
    }
}
