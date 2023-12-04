namespace Lokad.Onnx
{
    public static class CollectionExtensions
    {
        public static T? PeekIfNotEmpty<T>(this Stack<T> q) => q.Count > 0 ? q.Peek() : default(T);

        public static T? PopIfNotEmpty<T>(this Stack<T> q) => q.Count > 0 ? q.Pop() : default(T);

        public static void Pop<T>(this Stack<T> stack, int n)
        {
            for (int i = 1; i <= n; i++)
            {
                stack.Pop();
            }
        }
    }

    public static class CollectionUtils
    {
        public static string JoinWith(this IEnumerable<string> s, string j)
        {
            if (s.Count() == 0)
            {
                return "";
            }
            else if (s.Count() == 1)
            {
                return s.First();
            }
            else
            {
                return s.Aggregate((a, b) => a + j + b);
            }
        }

        public static string JoinWithSpaces(this IEnumerable<string> s)
        {
            if (s.Count() == 0)
            {
                return "";
            }
            else if (s.Count() == 1)
            {
                return s.First();
            }
            else
            {
                return s.Aggregate((a, b) => a + " " + b);
            }
        }

        public static T FailIfKeyNotPresent<T>(this Dictionary<string, object> d, string k)
            => d.ContainsKey(k) ? (T)d[k] : throw new KeyNotFoundException($"The required key {k} is not present.");

        public static T? TryGet<T>(this Dictionary<string, object> d, string k) => d.ContainsKey(k) ? (T)d[k] : default(T);

        public static T Get<T>(this Dictionary<string, object> d, string k) => d.ContainsKey(k) ? (T)d[k] : throw new KeyNotFoundException();

        public static void AddIfNotExists<K, V>(this Dictionary<K, V> d, K k, V v) where K : notnull
        {
            if (!d.ContainsKey(k))
            {
                d.Add(k, v);
            }
        }
    }
}