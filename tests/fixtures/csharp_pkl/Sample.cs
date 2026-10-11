using System;
using System.Collections.Generic;
using Alias = System.Text.StringBuilder;

namespace Shop.Orders
{
    public interface IRepository<T> : IDisposable
    {
        T Get(int id);
    }

    public abstract class BaseService
    {
        protected abstract void Run();
    }

    public class OrderService : BaseService, IRepository<Order>
    {
        private readonly ILogger _log;
        public int Count { get; set; }

        public OrderService(ILogger log) { _log = log; }
        ~OrderService() { }

        public Order Get(int id) => new Order(id);

        protected override void Run()
        {
            var o = Get(1);
            _log.Info("run");
            Helper.Static(o);
            Process<int>(2);
        }

        public async Task SaveAsync() { await Task.Delay(1); }
        public void Dispose() { }
    }

    public struct Point { public int X; }
    public record Order(int Id);
    public enum Status { New, Done }
    public delegate void Handler(object sender);
}

namespace Flat.Scoped;
static class Helper { public static void Static(object o) { } }
