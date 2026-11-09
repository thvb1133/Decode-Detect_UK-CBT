package dicegame;

// One box in the score table. It remembers the dice total even after
// the table prints X, so the end-of-game summary can still quote it.
public class ScoreCell {

    private final int total;
    private boolean struck;

    ScoreCell(int total) {
        this.total = total;
    }

    int getTotal() {
        return total;
    }

    boolean isStruck() {
        return struck;
    }

    void strike() {
        struck = true;
    }
}
