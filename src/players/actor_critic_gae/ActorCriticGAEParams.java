package players.actor_critic_gae;

import core.Types;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Java-side settings of the GAE actor-critic. The learning hyperparameters (hidden, gamma, alpha,
 * alpha_phi, beta, lam, episodes_per_iter, seed) live in python/config.yaml, since the networks
 * and training are in Python.
 *
 * Any field can be overridden by name (case-insensitive) with set("game_mode", "CAPITALS"); EnvServer's
 * KEY=VALUE arguments (passed by python/tribes_env.py) use this.
 */
@SuppressWarnings("WeakerAccess")
public class ActorCriticGAEParams {

    // ---------------------------------------------------------------- playing from Run (ActorCriticGAEPlayer)

    /** true: argmax over pi. false: sample from pi. */
    public boolean GREEDY = true;
    /** Checkpoint (python/train.py's .pkl) the player loads into python/serve.py. */
    public String MODEL_PATH = null;

    // ---------------------------------------------------------------- Tribes setup for training games

    /** The two tribes; the agent's seat (and so its tribe) is drawn at random every game. */
    public Types.TRIBE[] TRIBES = {Types.TRIBE.XIN_XI, Types.TRIBE.IMPERIUS};
    /** SCORE (30 turns, highest score wins) or CAPITALS. */
    public Types.GAME_MODE GAME_MODE = Types.GAME_MODE.SCORE;
    /** Level generator seed; -1 = a new random level every game. */
    public long LEVEL_SEED = -1;

    // ---------------------------------------------------------------- overrides

    /** Set a field from a string, e.g. set("lam", "0"), set("hidden", "128,128"), set("tribes", "BARDUR,OUMAJI"). */
    public void set(String key, String value) {
        Field f = find(key);
        if (f == null) throw new IllegalArgumentException("unknown parameter '" + key + "'; known: " + names());
        try {
            Class<?> t = f.getType();
            String v = value.trim();
            if (t == int.class) f.setInt(this, Integer.parseInt(v));
            else if (t == long.class) f.setLong(this, Long.parseLong(v));
            else if (t == double.class) f.setDouble(this, Double.parseDouble(v));
            else if (t == boolean.class) f.setBoolean(this, Boolean.parseBoolean(v));
            else if (t == String.class) f.set(this, v.isEmpty() || v.equals("null") ? null : v);
            else if (t == int[].class) f.set(this, Arrays.stream(v.split(",")).map(String::trim)
                    .filter(s -> !s.isEmpty()).mapToInt(Integer::parseInt).toArray());
            else if (t == Types.GAME_MODE.class) f.set(this, Types.GAME_MODE.valueOf(v.toUpperCase(Locale.ROOT)));
            else if (t == Types.TRIBE[].class) f.set(this, Arrays.stream(v.split(",")).map(String::trim)
                    .map(s -> Types.TRIBE.valueOf(s.toUpperCase(Locale.ROOT))).toArray(Types.TRIBE[]::new));
            else throw new IllegalArgumentException("can't parse type " + t + " for " + key);
        } catch (IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    public boolean has(String key) {
        return find(key) != null;
    }

    /** name -> value, for logging and for the run's config file. */
    public Map<String, String> asMap() {
        Map<String, String> m = new LinkedHashMap<>();
        for (Field f : getClass().getFields()) {
            if (Modifier.isStatic(f.getModifiers())) continue;
            try {
                Object v = f.get(this);
                String s;
                if (v instanceof int[]) s = Arrays.toString((int[]) v).replaceAll("[\\[\\] ]", "");
                else if (v instanceof Object[]) s = Arrays.toString((Object[]) v).replaceAll("[\\[\\] ]", "");
                else s = String.valueOf(v);
                m.put(f.getName().toLowerCase(Locale.ROOT), s);
            } catch (IllegalAccessException e) {
                throw new RuntimeException(e);
            }
        }
        return m;
    }

    @Override
    public String toString() {
        return asMap().toString();
    }

    private Field find(String key) {
        for (Field f : getClass().getFields())
            if (!Modifier.isStatic(f.getModifiers()) && f.getName().equalsIgnoreCase(key)) return f;
        return null;
    }

    private String names() {
        return asMap().keySet().toString();
    }
}
