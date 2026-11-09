package dicegame;

import java.util.ArrayList;
import java.util.NoSuchElementException;
import java.util.Random;
import java.util.Scanner;

// Plays eleven rounds at the console. The three players share the keyboard
// and take turns in the same order every round.
public class Game {

    static final int PLAYER_COUNT = 3;
    static final int ROUND_COUNT = 11;

    private final Scanner input;
    private final AppendableOut out;
    private final Dice dice;
    private final Player[] players;
    private final ScoreSheet sheet;

    Game(java.io.InputStream source, java.io.PrintStream destination, Random random) {
        this.input = new Scanner(source);
        this.out = new AppendableOut(destination);
        this.dice = new Dice(random);
        this.players = new Player[PLAYER_COUNT];
        for (int i = 0; i < PLAYER_COUNT; i++) {
            players[i] = new Player("Player " + (i + 1));
        }
        this.sheet = new ScoreSheet(PLAYER_COUNT);
    }

    void play() {
        printIntro();
        printTable();

        for (int round = 1; round <= ROUND_COUNT; round++) {
            String title = "Round " + round + " of " + ROUND_COUNT;
            out.println();
            out.println(title);
            out.println(underline(title.length()));

            // Round 1 is the only round where the three choices must differ.
            // Later rounds may reuse a column another player has already taken.
            boolean[] takenThisRound = new boolean[ScoreSheet.COLUMN_COUNT];
            for (int player = 0; player < PLAYER_COUNT; player++) {
                if (!takeTurn(round, player, takenThisRound)) {
                    return;
                }
            }
        }

        printEnding();
    }

    private boolean takeTurn(int round, int playerIndex, boolean[] takenThisRound) {
        Player player = players[playerIndex];
        out.println();
        out.println(player.getName() + "'s turn.");
        if (readLine("Press Enter to roll the dice. ") == null) {
            return false;
        }

        DiceThrow thrown = dice.roll();
        out.println();
        out.print(thrown.diagram());
        out.println("You rolled " + thrown.getFirst() + " and "
                + thrown.getSecond() + ". Total: " + thrown.getTotal() + ".");

        Integer column = chooseColumn(playerIndex, round, takenThisRound);
        if (column == null) {
            return false;
        }

        sheet.place(playerIndex, column, thrown.getTotal());
        takenThisRound[column - ScoreSheet.FIRST_COLUMN] = true;
        out.println(player.getName() + " places " + thrown.getTotal()
                + " in column " + column + ".");

        // A column can be completed in the middle of a round, as soon as
        // the third player chooses it. Score it then, not at the end.
        ColumnResult result = sheet.resolveIfComplete(column, players);
        if (result != null) {
            out.println(result.liveMessage(players));
        }
        printTable();
        return true;
    }

    // Keeps asking until the column is legal for this player and this round.
    // Returns null only when the console input runs out.
    private Integer chooseColumn(int playerIndex, int round, boolean[] takenThisRound) {
        while (true) {
            ArrayList<Integer> open = openColumns(playerIndex, round, takenThisRound);
            out.println("Columns still open to you: " + joinColumns(open));
            if (round == 1) {
                out.println("Round 1: each player must choose a different column.");
            }
            if (open.size() == 1) {
                out.println("This is the only column you have left.");
            }

            String line = readLine("Column: ");
            if (line == null) {
                return null;
            }
            line = line.trim();
            if (line.length() == 0) {
                out.println("Please enter a column number from 2 to 12.");
                continue;
            }

            int column;
            try {
                column = Integer.parseInt(line);
            } catch (NumberFormatException ex) {
                out.println("'" + line + "' is not a whole number. Enter a column from 2 to 12.");
                continue;
            }

            if (column < ScoreSheet.FIRST_COLUMN || column > ScoreSheet.LAST_COLUMN) {
                out.println("Column " + column + " is outside the table. Choose a column from 2 to 12.");
                continue;
            }
            if (sheet.isFilled(playerIndex, column)) {
                out.println("You have already used column " + column + ". Choose a different one.");
                continue;
            }
            if (round == 1 && takenThisRound[column - ScoreSheet.FIRST_COLUMN]) {
                out.println("Round 1 needs three different columns. Column "
                        + column + " was already chosen this round.");
                continue;
            }
            return column;
        }
    }

    private ArrayList<Integer> openColumns(int playerIndex, int round, boolean[] takenThisRound) {
        ArrayList<Integer> open = new ArrayList<Integer>();
        for (int column = ScoreSheet.FIRST_COLUMN; column <= ScoreSheet.LAST_COLUMN; column++) {
            if (sheet.isFilled(playerIndex, column)) {
                continue;
            }
            if (round == 1 && takenThisRound[column - ScoreSheet.FIRST_COLUMN]) {
                continue;
            }
            open.add(column);
        }
        return open;
    }

    private String joinColumns(ArrayList<Integer> columns) {
        if (columns.isEmpty()) {
            return "none";
        }
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < columns.size(); i++) {
            if (i > 0) {
                text.append(", ");
            }
            text.append(columns.get(i));
        }
        return text.toString();
    }

    private void printIntro() {
        out.println("Three-player dice game");
        out.println("======================");
        out.println("Two dice are thrown on each turn. Place the total in one");
        out.println("column from 2 to 12. You may use each column only once.");
        out.println("In round 1 the three players must pick different columns.");
        out.println("After that, a column may be shared.");
        out.println();
        out.println("When all three players have filled a column, the highest");
        out.println("total wins that column's points. A tie for the highest");
        out.println("total scores nothing. A lower total is shown as X.");
        out.println();
        out.println("There are 11 rounds. The highest score at the end wins.");
        out.println("Pass the keyboard along after each turn.");
    }

    private void printTable() {
        out.println();
        out.println("Score table");
        out.print(sheet.render(players));
        out.println("X marks a total beaten in that column. Scores are column points, not dice totals.");
    }

    private void printEnding() {
        out.println();
        out.println("Column summary");
        out.println("--------------");
        for (int column = ScoreSheet.FIRST_COLUMN; column <= ScoreSheet.LAST_COLUMN; column++) {
            ColumnResult result = sheet.getResult(column);
            if (result == null) {
                out.println("Column " + column + ": not complete, so it scores nothing.");
                continue;
            }
            out.println(result.summaryLine(players));
        }

        out.println();
        out.println("Final scores");
        out.println("------------");
        int best = players[0].getScore();
        for (int i = 1; i < players.length; i++) {
            if (players[i].getScore() > best) {
                best = players[i].getScore();
            }
        }

        int leaders = 0;
        for (int i = 0; i < players.length; i++) {
            out.println(players[i].getName() + ": " + players[i].getScore());
            if (players[i].getScore() == best) {
                leaders++;
            }
        }

        out.println();
        if (leaders == 1) {
            String winner = "";
            for (int i = 0; i < players.length; i++) {
                if (players[i].getScore() == best) {
                    winner = players[i].getName();
                }
            }
            out.println("Result: " + winner + " wins the game with " + best + " points.");
        } else {
            out.println("Result: the game is a tie between "
                    + leaderNames(best) + ", all on " + best + " points.");
        }
    }

    private String underline(int length) {
        StringBuilder line = new StringBuilder();
        for (int i = 0; i < length; i++) {
            line.append('-');
        }
        return line.toString();
    }

    private String leaderNames(int best) {
        String[] names = new String[players.length];
        int count = 0;
        for (int i = 0; i < players.length; i++) {
            if (players[i].getScore() == best) {
                names[count] = players[i].getName();
                count++;
            }
        }
        if (count == 2) {
            return names[0] + " and " + names[1];
        }
        return names[0] + ", " + names[1] + " and " + names[2];
    }

    private String readLine(String prompt) {
        out.print(prompt);
        out.flush();
        try {
            if (!input.hasNextLine()) {
                out.println();
                out.println("Input has ended, so the game has stopped.");
                return null;
            }
            return input.nextLine();
        } catch (NoSuchElementException ex) {
            out.println();
            out.println("Input has ended, so the game has stopped.");
            return null;
        } catch (IllegalStateException ex) {
            out.println();
            out.println("Input has ended, so the game has stopped.");
            return null;
        }
    }

    // Small wrapper so a prompt can be printed without a newline and flushed.
    private static class AppendableOut {
        private final java.io.PrintStream destination;

        AppendableOut(java.io.PrintStream destination) {
            this.destination = destination;
        }

        void println() {
            destination.println();
            destination.flush();
        }

        void println(String line) {
            destination.println(line);
            destination.flush();
        }

        void print(String text) {
            destination.print(text);
            destination.flush();
        }

        void flush() {
            destination.flush();
        }
    }
}
